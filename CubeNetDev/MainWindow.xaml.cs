using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using LiveCharts;
using LiveCharts.Defaults;
using Warp;
using Warp.Tools;
using Warp.NNModels;

namespace CubeNetDev
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private bool NeedsExport = false;

        public MainWindow()
        {
            InitializeComponent();

            var Options = ((App)Application.Current).Options;

            int[] GPUsTrain = Options.GPUNetwork.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(s => int.Parse(s)).ToArray();

            CubeNetTorch Network = null;

            Task.Run(() =>
            {
                if (Options.Mode == "train" || Options.Mode == "both")
                {
                    Dispatcher.Invoke(() => TextProgress.Text = "Loading examples...");

                    GPU.SetDevice(Options.GPUPreprocess);

                    List<float[]> AllVolumes = new List<float[]>();
                    List<float[]> AllLabels = new List<float[]>();
                    List<int3> AllDims = new List<int3>();
                    List<List<int3>> AllLabelPositions = new List<List<int3>>();
                    int3 DimsLargest = new int3(1);
                    float[] LabelWeights = Helper.ArrayOfConstant(1f, Options.NLabels);

                    long[] ClassHist = new long[2];

                    bool DoingParticles = !string.IsNullOrEmpty(Options.LabelsCoordsPath);

                    List<string> PathsVolumes = Directory.EnumerateFiles(Path.Combine(Options.WorkingDirectory, Options.VolumesPath), "*.mrc").ToList();

                    if (DoingParticles)
                        PathsVolumes.RemoveAll(p => !File.Exists(Path.Combine(Options.WorkingDirectory, Options.LabelsCoordsPath, Helper.PathToName(p) + Options.LabelsSuffix + ".star")));
                    else
                        PathsVolumes.RemoveAll(p => !File.Exists(Path.Combine(Options.WorkingDirectory, Options.LabelsVolumePath, Helper.PathToName(p) + Options.LabelsSuffix + ".mrc")));

                    for (int p = 0; p < PathsVolumes.Count; p++)
                    {
                        Image ExampleVolume = Image.FromFile(PathsVolumes[p]);
                        ExampleVolume.Normalize(true);
                        ExampleVolume.FreeDevice();
                        AllVolumes.Add(ExampleVolume.GetHostContinuousCopy());

                        int3 Dims = ExampleVolume.Dims;
                        AllDims.Add(Dims);

                        Image LabelsVolume = null;
                        {
                            if (DoingParticles)
                            {
                                Star TableIn = new Star(Path.Combine(Options.WorkingDirectory, Options.LabelsCoordsPath, Helper.PathToName(PathsVolumes[p]) + Options.LabelsSuffix + ".star"));
                                int3[] Coords = TableIn.GetRelionCoordinates().Select(v => new int3(v * new float3(Dims) + new float3(0.5f))).ToArray();

                                LabelsVolume = new Image(Dims);

                                foreach (var coord in Coords)
                                {
                                    LabelsVolume.TransformRegionValues(new int3(Options.LabelsRadius * 2 + 1), coord, (pos, posCentered, v) =>
                                    {
                                        return posCentered.Length() < Options.LabelsRadius + 1e-5f ? 1 : 0;
                                    });
                                }

                                if (!string.IsNullOrEmpty(Options.DiagnosticsPath))
                                {
                                    Directory.CreateDirectory(Path.Combine(Options.WorkingDirectory, Options.DiagnosticsPath));
                                    LabelsVolume.WriteMRC(Path.Combine(Options.WorkingDirectory, Options.DiagnosticsPath, Helper.PathToNameWithExtension(PathsVolumes[p])), true);
                                }
                            }
                            else
                            {
                                LabelsVolume = Image.FromFile(Path.Combine(Options.WorkingDirectory, Options.LabelsVolumePath, Helper.PathToName(PathsVolumes[p]) + Options.LabelsSuffix + ".mrc"));
                            }
                        }
                        AllLabels.Add(LabelsVolume.GetHostContinuousCopy());

                        float[][] LabelsData = LabelsVolume.GetHost(Intent.Read);
                        List<int3> LabelPositions = new List<int3>();

                        {
                            for (int z = 0; z < LabelsData.Length; z++)
                                for (int i = 0; i < LabelsData[z].Length; i++)
                                {
                                    int x = i % LabelsVolume.Dims.X;
                                    int y = i / LabelsVolume.Dims.X;

                                    int Label = Math.Min(1, (int)(LabelsData[z][i] + 0.5f));

                                    ClassHist[Label]++;
                                    if (Label > 0)
                                        LabelPositions.Add(new int3(x, y, z));
                                }
                        }

                        AllLabelPositions.Add(LabelPositions);

                        DimsLargest = int3.Max(DimsLargest, Dims);
                    }

                    {
                        LabelWeights[0] = 0.05f;// (float)Math.Pow((float)ClassHist.Skip(1).Sum() / ClassHist[0], 1 / 3f);
                                            //LabelWeights[1] = 1;
                    }

                    GPU.SetDevice(Options.GPUPreprocess);

                    Dispatcher.Invoke(() => TextProgress.Text = "Loading model...");

                    Network = new CubeNetTorch(Options.NLabels, new int3(Options.WindowSize), LabelWeights, GPUsTrain, Options.BatchSize);
                    if (!string.IsNullOrEmpty(Options.OldModelName))
                        Network.Load(Path.Combine(Options.WorkingDirectory, Options.OldModelName));

                    #region Training

                    Dispatcher.Invoke(() => TextProgress.Text = "Training...");

                    int3 DimsAugmented = Network.BoxDimensions;
                    int Border = (CubeNet.BoxDimensionsTrain.X - CubeNet.BoxDimensionsValidTrain.X) / 2;
                    int BatchSize = Network.BatchSize;
                    int PlotEveryN = 10;

                    int ElementsSliceRaw = (int)DimsAugmented.Elements();
                    int ElementsBatchRaw = ElementsSliceRaw * BatchSize;

                    List<ObservablePoint> TrainLossPoints = new List<ObservablePoint>();
                    Queue<float> LastTrainLosses = new Queue<float>(PlotEveryN);

                    GPU.SetDevice(1);

                    Image d_Volume = new Image(IntPtr.Zero, DimsLargest);
                    Image d_Labels = new Image(IntPtr.Zero, DimsLargest);

                    Image d_AugmentedData = new Image(IntPtr.Zero, new int3(DimsAugmented.X, DimsAugmented.Y, DimsAugmented.Z * BatchSize));
                    Image d_AugmentedLabels = new Image(IntPtr.Zero, new int3(DimsAugmented.X, DimsAugmented.Y, DimsAugmented.Z * Options.NLabels * BatchSize));

                    Stopwatch Watch = new Stopwatch();
                    Watch.Start();

                    Random RG = new Random(123);
                    RandomNormal RGN = new RandomNormal(123);

                    int NDone = 0, NDoneAll = 0;
                    for (int b = 0; b < Options.NIterations; b++)
                    {
                        int ExampleID = RG.Next(AllVolumes.Count);
                        int3 Dims = AllDims[ExampleID];

                        List<int3> LabelPositions = AllLabelPositions[ExampleID];

                        float3[] Translations = Helper.ArrayOfFunction(x =>
                        {
                            bool SampleLabels = (b * BatchSize + x) % (Options.OversampleBackground + 1) == 0 && LabelPositions.Count > 0;

                            int3 Position;

                            if (SampleLabels)
                            {
                                Position = LabelPositions[RG.Next(LabelPositions.Count)];
                                Position += new int3(RG.Next(DimsAugmented.X / 2), RG.Next(DimsAugmented.Y / 2), RG.Next(DimsAugmented.Z / 2)) - new int3(DimsAugmented / 4);
                            }
                            else
                            {
                                Position = new int3(RG.Next(Dims.X - DimsAugmented.X), RG.Next(Dims.Y - DimsAugmented.Y), RG.Next(Dims.Z - DimsAugmented.Z)) + DimsAugmented / 2;
                            }

                            Position = int3.Max(int3.Min(Position, Dims - DimsAugmented / 2), DimsAugmented / 2);

                            return new float3(Position.X, Position.Y, Position.Z);
                        }, BatchSize);

                        float3[] Rotations = Helper.ArrayOfFunction(i => new float3((float)(RG.Next(2) * Math.PI),
                                                                                    (float)(RG.Next(2) * Math.PI),
                                                                                    (float)(RG.Next(2) * Math.PI)), BatchSize);
                        float3[] Scales = Helper.ArrayOfFunction(i => new float3(1,
                                                                                1,
                                                                                0), BatchSize);
                        float StdDev = 0;// (float)Math.Abs(RGN[threadID].NextSingle(0, 0f));

                        d_AugmentedLabels.Fill(0);

                        GPU.CopyHostToDevice(AllVolumes[ExampleID], d_Volume.GetDevice(Intent.Write), AllVolumes[ExampleID].Length);
                        GPU.CopyHostToDevice(AllLabels[ExampleID], d_Labels.GetDevice(Intent.Write), AllLabels[ExampleID].Length);

                        GPU.CubeNetAugment(d_Volume.GetDevice(Intent.Read),
                                            d_Labels.GetDevice(Intent.Read),
                                            Dims,
                                            d_AugmentedData.GetDevice(Intent.Write),
                                            d_AugmentedLabels.GetDevice(Intent.Write),
                                            Options.NLabels,
                                            DimsAugmented,
                                            Helper.ToInterleaved(Translations),
                                            Helper.ToInterleaved(Rotations),
                                            Helper.ToInterleaved(Scales),
                                            StdDev,
                                            RG.Next(99999999),
                                            (uint)BatchSize);


                        float LearningRate = (float)Math.Exp(MathHelper.Lerp((float)Math.Log(1e-3), (float)Math.Log(1e-5), (float)NDone / Options.NIterations));
                        //Dispatcher.Invoke(() => LearningRate = (float)SliderLearningRate.Value);
                        Dispatcher.Invoke(() => SliderLearningRate.Value = (decimal)LearningRate);

                        Image ResultPrediction;
                        float[] ResultLoss;

                        Network.Train(d_AugmentedData,
                                      d_AugmentedLabels,
                                      LearningRate,
                                      NDone % PlotEveryN == 0,
                                      out ResultPrediction,
                                      out ResultLoss);


                        {

                            //if (PositionsGround.Length > 0)
                            //{
                            //    LastTrainAccuracies.Enqueue(BatchTrainAccuracy);
                            //    if (LastTrainAccuracies.Count > PlotEveryN)
                            //        LastTrainAccuracies.Dequeue();
                            //}

                            LastTrainLosses.Enqueue(ResultLoss[0]);
                            if (LastTrainLosses.Count > PlotEveryN)
                                LastTrainLosses.Dequeue();

                            if (NDone % PlotEveryN == 0)
                            {
                                TrainLossPoints.Add(new ObservablePoint((float)NDone,
                                                                        MathHelper.Mean(LastTrainLosses)));


                                //long Elapsed = Watch.ElapsedMilliseconds;
                                //double Estimated = (double)Elapsed / NDone * AllBatches.Count;
                                //int Remaining = (int)(Estimated - Elapsed);
                                //TimeSpan SpanRemaining = new TimeSpan(0, 0, 0, 0, Remaining);

                                Func<float[], int2, ImageSource> MakeImage = (data, dims) =>
                                {
                                    byte[] Bytes = new byte[dims.Elements()];
                                    for (int i = 0; i < data.Length; i++)
                                        Bytes[i] = (byte)(Math.Max(0, Math.Min(1, data[i])) * 255);

                                    ImageSource SliceImage = BitmapSource.Create(dims.X, dims.Y, 96, 96, PixelFormats.Indexed8, BitmapPalettes.Gray256, Bytes, dims.X);
                                    SliceImage.Freeze();

                                    return SliceImage;
                                };

                                Dispatcher.Invoke(() =>
                                {
                                //SeriesTrainAccuracy.Values = new ChartValues<ObservablePoint>(TrainAccuracyPoints);
                                SeriesLoss.Values = new ChartValues<ObservablePoint>(TrainLossPoints.Skip(Math.Max(0, TrainLossPoints.Count - 100000)));

                                    {
                                        int NSlices = 5;
                                        int SliceElements = (int)DimsAugmented.Slice().Elements();
                                        float[] Slices = new float[SliceElements * NSlices];
                                        GPU.CopyDeviceToHost(d_AugmentedData.GetDeviceSlice(DimsAugmented.Z / 2 - (NSlices / 2), Intent.Read), Slices, Slices.Length);

                                        float[] Averaged = new float[SliceElements];
                                        for (int s = 0; s < NSlices; s++)
                                            for (int i = 0; i < SliceElements; i++)
                                                Averaged[i] += Slices[s * SliceElements + i];

                                        float2 MeanStd = MathHelper.MeanAndStd(Averaged);
                                        for (int i = 0; i < Averaged.Length; i++)
                                            Averaged[i] = ((Averaged[i] - MeanStd.X) / MeanStd.Y + 2.5f) / 5;

                                        ImageInput.Source = MakeImage(Averaged, new int2(DimsAugmented));
                                    }
                                    {
                                        int NSlices = 9;
                                        int SliceElements = (int)DimsAugmented.Slice().Elements();
                                        float[] Slices = new float[SliceElements * NSlices];
                                        GPU.CopyDeviceToHost(d_AugmentedLabels.GetDeviceSlice(DimsAugmented.Z + DimsAugmented.Z / 2 - (NSlices / 2), Intent.Read), Slices, Slices.Length); // One-hot vector, skip first

                                        float[] Averaged = new float[SliceElements];
                                        for (int s = 0; s < NSlices; s++)
                                            for (int i = 0; i < SliceElements; i++)
                                                Averaged[i] = Math.Max(Slices[s * SliceElements + i], Averaged[i]);

                                        if (Options.NLabels > 2)
                                            for (int i = 0; i < Averaged.Length; i++)
                                                Averaged[i] /= Options.NLabels - 1;

                                        ImageLabels.Source = MakeImage(Averaged, new int2(DimsAugmented));
                                    }
                                    {
                                        int NSlices = 9;
                                        int SliceElements = (int)DimsAugmented.Slice().Elements();
                                        float[] Slices = new float[SliceElements * NSlices];
                                        GPU.CopyDeviceToHost(ResultPrediction.GetDeviceSlice(DimsAugmented.Z / 2 - (NSlices / 2), Intent.Read), Slices, Slices.Length);

                                        float[] Averaged = new float[SliceElements];
                                        for (int s = 0; s < NSlices; s++)
                                            for (int i = 0; i < SliceElements; i++)
                                                Averaged[i] = Math.Max(Slices[s * SliceElements + i], Averaged[i]);

                                        if (Options.NLabels > 2)
                                            for (int i = 0; i < Averaged.Length; i++)
                                                Averaged[i] /= Options.NLabels - 1;

                                        ImagePrediction.Source = MakeImage(Averaged, new int2(DimsAugmented));
                                    }
                                });
                            }

                            if (NeedsExport || Watch.Elapsed.TotalHours > 1 || NDone + 1 >= Options.NIterations)
                            {
                                NeedsExport = false;
                                Watch.Restart();

                                Network.Save(Path.Combine(Options.WorkingDirectory, "CubeNet" + DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".pt"));
                                Thread.Sleep(10000);

                                Dispatcher.Invoke(() => ButtonExport.IsEnabled = true);
                            }
                        }


                        NDone++;
                    }

                    d_Volume.Dispose();
                    d_Labels.Dispose();

                    d_AugmentedData.Dispose();
                    d_AugmentedLabels.Dispose();

                    TorchSharp.Torch.CudaEmptyCache();

                    #endregion
                }

                if (Options.Mode == "infer" || Options.Mode == "both")
                {
                    if (Network == null)
                    {
                        Network = new CubeNetTorch(Options.NLabels, new int3(Options.WindowSize), Helper.ArrayOfConstant(1f, Options.NLabels), GPUsTrain, Options.BatchSize);
                        if (!string.IsNullOrEmpty(Options.OldModelName))
                            Network.Load(Path.Combine(Options.WorkingDirectory, Options.OldModelName));
                    }

                    List<string> PathsVolumes = Directory.EnumerateFiles(Path.Combine(Options.WorkingDirectory, Options.VolumesPath), "*.mrc").ToList();

                    foreach (var pathVolume in PathsVolumes)
                    {
                        Image Volume = Image.FromFile(pathVolume);
                        Volume.Normalize(true);
                        Volume.FreeDevice();

                        Image Segmentation = Network.Segment(Volume, Options.PredictionThreshold);
                        Volume.Dispose();

                        float4[] Components = Network.Match(Segmentation, Options.ParticleDiameter);
                        Components = Components.Where(c => c.W >= Options.MinimumVoxels).ToArray();

                        float3[] Coords = Components.Select(c => new float3(c.X, c.Y, c.Z) / new float3(Segmentation.Dims)).ToArray();
                        float4[] CoordsScored = Coords.Select((v, i) => new float4(v.X, v.Y, v.Z, Components[i].W)).ToArray();

                        Directory.CreateDirectory(Path.Combine(Options.WorkingDirectory, Options.OutputPath));
                        Segmentation.WriteMRC(Path.Combine(Options.WorkingDirectory, Options.OutputPath, Helper.PathToName(pathVolume) + "_seg.mrc"), true);

                        Star TableOut = new Star(CoordsScored, "rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnAutopickFigureOfMerit");
                        TableOut.Save(Path.Combine(Options.WorkingDirectory, Options.OutputPath, Helper.PathToName(pathVolume) + "_particles.star"));

                        Segmentation.Dispose();
                    }
                }
            });
        }

        private void ButtonExport_OnClick(object sender, RoutedEventArgs e)
        {
            NeedsExport = true;
            ButtonExport.IsEnabled = false;
        }
    }
}
