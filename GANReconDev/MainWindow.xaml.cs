using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls.Primitives;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using LiveCharts;
using LiveCharts.Defaults;
using TorchSharp;
using TorchSharp.NN;
using TorchSharp.Tensor;
using Warp;
using Warp.Headers;
using Warp.NNModels;
using Warp.Tools;

namespace ParticleWGANDev
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public decimal LearningRate
        {
            get { return (decimal)GetValue(LearningRateProperty); }
            set { SetValue(LearningRateProperty, value); }
        }

        public static readonly DependencyProperty LearningRateProperty = DependencyProperty.Register("LearningRate", typeof(decimal), typeof(MainWindow), new PropertyMetadata(0.001M));

        public struct StartUpSettings
        {
            public int BatchSize;
            public decimal LearningRate;
            public decimal Reduction;
            public float Lambda;
            public int DiscIters;
            public string LogFileName;
            public string OutDirectory;
            public StartUpSettings(int batchSize, decimal learningRate, decimal reduction, float lambda, int discIters, string logFileName, string outDirectory)
            {
                BatchSize = batchSize;
                LearningRate = learningRate;
                Reduction = reduction;
                Lambda = lambda;
                DiscIters = discIters;
                LogFileName = logFileName;
                OutDirectory = outDirectory;
            }
        }

        public static StartUpSettings Settings = new(16, 0.001M, 0.9M, 0.01f, 8, "log.txt", @"D:\GAN_recon_polcompl\");

        private void CheckSaveRecs_Checked(object sender, RoutedEventArgs e) { ShouldSaveRecs = true; }
        private void CheckSaveRecs_Unchecked(object sender, RoutedEventArgs e) { ShouldSaveRecs = false; }

        private void CheckTrainGen_Checked(object sender, RoutedEventArgs e) { ShouldTrainOnlyGen = true; }
        private void CheckTrainGen_Unchecked(object sender, RoutedEventArgs e) { ShouldTrainOnlyGen = false; }

        private bool ShouldSaveModel = false;
        private bool ShouldSaveRecs = false;
        private bool ShouldTrainOnlyGen = false;

        CancellationTokenSource cancellationTokenSourceWindow = new CancellationTokenSource();

        struct ThreadArgs
        {
            public CancellationToken Token;
            public int ThreadId;

            public ThreadArgs(CancellationToken token, int threadId)
            {
                Token = token;
                ThreadId = threadId;
            }
        }

        private string WorkingDirectory = @"D:\GAN_recon_polcompl\";
        private string OutDirectory = @"D:\GAN_recon_polcompl\";
        private string DirectoryReal = "particles";
        private string DirectoryFake = "sim";
        const int numEpochs = 100;
        const int DimGenerator = 64;
        const int DimZoom = 128;
        decimal reduction = 0.9M;
        const double sigmaShiftPix = 0.5;
        const double sigmaShiftRel = sigmaShiftPix / (DimZoom / 2);
        private double LowPass = 1.0;

        private int BatchSize = 32;
        float Lambda = 0.6f;
        int DiscIters = 5;
        bool TrainGen = true;

        int NThreads = 1;
        int PreProcessingDevice = 1;
        int ProcessingDevice = 0;
        string logFileName = "log.txt";
        public MainWindow()
        {
            InitializeComponent();
            BatchSize = Settings.BatchSize;
            LearningRate = Settings.LearningRate;
            reduction = Settings.Reduction;
            Lambda = Settings.Lambda;
            DiscIters = Settings.DiscIters;

            OutDirectory = $@"{Settings.OutDirectory}\";
            if (!Directory.Exists(OutDirectory))
            {
                _ = Directory.CreateDirectory(OutDirectory);
            }
            
            logFileName = $@"{Settings.OutDirectory}\{Settings.LogFileName}";
            //doTraining();
            SliderLearningRate.DataContext = this;
            ButtonStartParticle.IsEnabled = false;
            Task.Run(doTraining, cancellationTokenSourceWindow.Token);
        }
        private void CloseMainwindow()
        {
            cancellationTokenSourceWindow.Cancel();
            Application.Current.Dispatcher.Invoke(() => Application.Current.Shutdown());
        }

        private static float getGaussian(Random rand, double mu, double sigma)
        {
            double u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            return (float)(mu + sigma * randStdNormal);
        }

        private void doTraining() {
            int seed = 42;
            WriteToLog("Loading model... (" + GPU.GetFreeMemory(ProcessingDevice) + " MB free)");
            Torch.SetSeed(seed);
            //ParticleWGAN TrainModel = new ParticleWGAN(new int2(DimGenrator), 32, new[] { 1 }, BatchSize);
            //Image refVolume = Image.FromFile(Path.Combine(WorkingDirectory, "run_1k_unfil.mrc")).AsScaled(new int3(DimGenrator));
            ReconstructionWGAN TrainModel = new ReconstructionWGAN(new int2(DimGenerator), new[] { ProcessingDevice }, BatchSize);
            TrainModel.SigmaShift = sigmaShiftRel;
            //TrainModel.Load(@"D:\GAN_recon_polcompl\ParticleWGAN_SN_20210910_161349.pt");
            WriteToLog("Done. (" + GPU.GetFreeMemory(ProcessingDevice) + " MB free)");

            GPU.SetDevice(PreProcessingDevice);

            Image[] ImagesReal = Helper.ArrayOfFunction(i => new Image(new int3(DimGenerator, DimGenerator, BatchSize)), DiscIters + 1);
            Image[] ImagesCTF = Helper.ArrayOfFunction(i => new Image(new int3(DimGenerator, DimGenerator, BatchSize), true), DiscIters + 1);
            Image[] ImagesNoise = Helper.ArrayOfFunction(i => new Image(new int3(DimGenerator, DimGenerator, BatchSize)), DiscIters + 1);
            float[][] ImagesAngles = Helper.ArrayOfFunction(i => new float[BatchSize * 3], DiscIters + 1);

            Semaphore ReloadBlock = new Semaphore(1, 1);
            bool HasBeenProcessed = true;

            Star TableIn = new Star(Path.Combine(WorkingDirectory, "cryosparc_P243_J525_003_particles.star"));

            Random rand = new Random(seed);


            string[] ColumnStackNames = TableIn.GetColumn("rlnImageName").Select(s => Helper.PathToNameWithExtension(s.Substring(s.IndexOf('@') + 1))).ToArray();
            HashSet<string> UniqueStackNames = Helper.GetUniqueElements(ColumnStackNames);
            //UniqueStackNames.RemoveWhere(s => !File.Exists(Path.Combine(WorkingDirectory, DirectoryReal, s)));
            //int[] KeepRows = Helper.ArrayOfSequence(0, TableIn.RowCount, 1).Where(r => UniqueStackNames.Contains(ColumnStackNames[r])).ToArray();
            //TableIn = TableIn.CreateSubset(KeepRows);

            int DimRaw = MapHeader.ReadFromFile(Path.Combine(WorkingDirectory, "Refine3D_CryoSparcSelected_run_class001.mrc")).Dimensions.X;

            var AllParticleAddresses = new (int id, string name)[TableIn.RowCount];
            /*
            {
                ColumnStackNames = TableIn.GetColumn("rlnImageName");
                for (int r = 0; r < TableIn.RowCount; r++)
                {
                    string s = ColumnStackNames[r];
                    int ID = int.Parse(s.Substring(0, s.IndexOf('@'))) - 1;
                    string Name = Helper.PathToNameWithExtension(s.Substring(s.IndexOf('@') + 1));
                    AllParticleAddresses[r] = (ID, Name);
                }
            }
            */
            int[] AllIDs = Helper.ArrayOfSequence(0, AllParticleAddresses.Length, 1);

            CTF[] AllParticleCTF = TableIn.GetRelionCTF();
            float3[] AllParticleAngles = TableIn.GetRelionAngles().Select(s => s * Helper.ToRad).ToArray();

            float3[] RandomParticleAngles = Helper.GetHealpixAngles(3).Select(s => s * Helper.ToRad).ToArray();

            int numParticles = RandomParticleAngles.Length;
            int currentEpoch = 0;

            string starFileName = @$"{WorkingDirectory}\run_model.star";
            float[][] AllSigmas = Helper.ArrayOfFunction(i => {
                Star table = new(starFileName, $"model_group_{i + 1}");
                string[] column = table.GetColumn("rlnSigma2Noise");
                float[] entries = column.Select(s => (float)Math.Sqrt(float.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture))).ToArray();
                return entries;
            }, 1);

            ParameterizedThreadStart ReloadLambda = (par) =>
            {
                ThreadArgs args = (ThreadArgs)par;
                GPU.SetDevice(PreProcessingDevice);
                Image TrefVolume = Image.FromFile(Path.Combine(WorkingDirectory, "Refine3D_CryoSparcSelected_run_class001.mrc"));
                if (DimZoom != DimRaw)
                    TrefVolume = TrefVolume.AsRegion(new int3((DimRaw - DimZoom) / 2), new int3(DimZoom));
                Projector TProj = new Projector(TrefVolume, 2);

                Random ReloadRand = new Random(args.ThreadId);
                RandomNormal NoiseRand = new RandomNormal(args.ThreadId);
                Random ShiftRand = new Random(args.ThreadId);
                Random sigmaPicker = new Random(args.ThreadId);

                bool OwnBatchUsed = true;
                Image LoadStack = new Image(new int3(DimRaw, DimRaw, BatchSize));

                Image[] TImagesReal = Helper.ArrayOfFunction(i => new Image(new int3(DimGenerator, DimGenerator, BatchSize)), DiscIters + 1);
                Image[] TImagesNoise = Helper.ArrayOfFunction(i => new Image(new int3(DimGenerator, DimGenerator, BatchSize)), DiscIters + 1);
                Image[] TImagesCTFFull = Helper.ArrayOfFunction(i => { Image im = new Image(new int3(DimZoom, DimZoom, BatchSize), true); im.Fill(1); return im; }, DiscIters + 1);
                Image[] TImagesCTFScaled = Helper.ArrayOfFunction(i => { Image im = new Image(new int3(DimGenerator, DimGenerator, BatchSize), true); im.Fill(1); return im; }, DiscIters + 1);

                float[][] TImagesAngles = Helper.ArrayOfFunction(i => new float[BatchSize * 3], DiscIters + 1);
                Image CTFCoordsFull = CTF.GetCTFCoords(DimZoom, DimZoom);
                Image CTFCoordsScaled = CTF.GetCTFCoords(DimGenerator, DimZoom);
                float2[][] scaledCoords = CTFCoordsScaled.GetHostComplexCopy();
                float2[][] FullCoords = CTFCoordsFull.GetHostComplexCopy();

                Image CTFMaskFull = new Image(new int3(DimZoom, DimZoom, BatchSize), true);
                CTFMaskFull.Fill(1);
                CTFMaskFull.TransformValues((x, y, z, val) =>
                {
                    float nyquistsoftedge = 0.05f;
                    float yy = y >= CTFMaskFull.Dims.Y / 2 + 1 ? y - CTFMaskFull.Dims.Y : y;
                    yy /= CTFMaskFull.Dims.Y / 2.0f;
                    yy *= yy;


                    float xx = x;
                    xx /= CTFMaskFull.Dims.X / 2.0f;
                    xx *= xx;

                    float r = (float)Math.Sqrt(xx + yy);

                    float filter = 1;
                    if (nyquistsoftedge > 0)
                    {
                        float edgelow = (float)Math.Cos(Math.Min(1, Math.Max(0, 0 - r) / nyquistsoftedge) * Math.PI) * 0.5f + 0.5f;
                        float edgehigh = (float)Math.Cos(Math.Min(1, Math.Max(0, (r - 1) / nyquistsoftedge)) * Math.PI) * 0.5f + 0.5f;
                        filter = edgelow * edgehigh;
                    }
                    else
                    {
                        filter = (r is >= 0 and <= 1) ? 1 : 0;
                    }

                    return val * filter;
                });
                Image CTFMaskScaled = new Image(new int3(DimGenerator, DimGenerator, BatchSize), true);
                CTFMaskScaled.Fill(1);
                CTFMaskScaled.TransformValues((x, y, z, val) =>
                {
                    float nyquistsoftedge = 0.05f;
                    float yy = y >= CTFMaskScaled.Dims.Y / 2 + 1 ? y - CTFMaskScaled.Dims.Y : y;
                    yy /= CTFMaskScaled.Dims.Y / 2.0f;
                    yy *= yy;


                    float xx = x;
                    xx /= CTFMaskScaled.Dims.X / 2.0f;
                    xx *= xx;

                    float r = (float)Math.Sqrt(xx + yy);

                    float filter = 1;
                    if (nyquistsoftedge > 0)
                    {
                        float edgelow = (float)Math.Cos(Math.Min(1, Math.Max(0, 0 - r) / nyquistsoftedge) * Math.PI) * 0.5f + 0.5f;
                        float edgehigh = (float)Math.Cos(Math.Min(1, Math.Max(0, (r - 1) / nyquistsoftedge)) * Math.PI) * 0.5f + 0.5f;
                        filter = edgelow * edgehigh;
                    }
                    else
                    {
                        filter = (r is >= 0 and <= 1) ? 1 : 0;
                    }

                    return val * filter;
                });
                

                int PlanForw = 0, PlanBack = 0;
                if (DimRaw != DimGenerator)
                {
                    PlanForw = GPU.CreateFFTPlan(new int3(DimRaw, DimRaw, 1), (uint)BatchSize);
                    PlanBack = GPU.CreateIFFTPlan(new int3(DimGenerator, DimGenerator, 1), (uint)BatchSize);
                }

                while (!args.Token.IsCancellationRequested)
                {
                    // If this thread succeeded at pushing its previously loaded batch to processing
                    if (OwnBatchUsed)
                    {
                        for (int iterTrain = 0; iterTrain < DiscIters + 1; iterTrain++)
                        {
                            int[] SubsetIDs = Helper.RandomSubset(AllIDs, BatchSize, ReloadRand.Next());
                            int[] AngleIds = Helper.ArrayOfFunction(i => ReloadRand.Next(0, RandomParticleAngles.Length), BatchSize);
                            int[] SigmaIds = Helper.ArrayOfFunction(i => sigmaPicker.Next(0, AllSigmas.Length), BatchSize);

                            // Read, and copy or rescale real and fake images from prepared stacks

                            float3[] theseAngles = Helper.IndexedSubset(RandomParticleAngles, AngleIds);
                            float[][] theseSigmas = Helper.IndexedSubset(AllSigmas, SigmaIds);
                            TImagesAngles[iterTrain] = Helper.ToInterleaved(theseAngles);

                            Image projected = TProj.ProjectToRealspace(new int2(DimZoom), theseAngles);

                            float3[] shiftsPix = Helper.ArrayOfFunction(i =>
                            {
                                float x, y;
                                {
                                    double u1 = 1.0 - ShiftRand.NextDouble(); //uniform(0,1] random doubles
                                    double u2 = 1.0 - ShiftRand.NextDouble();
                                    double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                                 Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
                                    x = (float)(0 + sigmaShiftPix * randStdNormal);
                                }
                                {
                                    double u1 = 1.0 - ShiftRand.NextDouble(); //uniform(0,1] random doubles
                                    double u2 = 1.0 - ShiftRand.NextDouble();
                                    double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                                 Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
                                    y = (float)(0 + sigmaShiftPix * randStdNormal);
                                }
                                return new float3(x, y, 0);
                            }, projected.Dims.Z);

                            //projected.ShiftSlices(shiftsPix);

                            float3[] shiftsRel = Helper.ArrayOfFunction(i => shiftsPix[i] * 1.0f / (DimZoom / 2), shiftsPix.Length);


                            GPU.CheckGPUExceptions();
                            var theseCTF = Helper.IndexedSubset(AllParticleCTF, SubsetIDs).Select(c => c.ToStruct()).ToArray();
                            GPU.CreateCTF(TImagesCTFFull[iterTrain].GetDevice(Intent.Write),
                                          CTFCoordsFull.GetDevice(Intent.Read),
                                          IntPtr.Zero,
                                          (uint)CTFCoordsFull.ElementsSliceComplex,
                                          theseCTF,
                                          false,
                                          (uint)BatchSize);
                            GPU.CreateCTF(TImagesCTFScaled[iterTrain].GetDevice(Intent.Write),
                              CTFCoordsScaled.GetDevice(Intent.Read),
                              IntPtr.Zero,
                              (uint)CTFCoordsScaled.ElementsSliceComplex,
                              theseCTF,
                              false,
                              (uint)BatchSize);

                            {
                                Image thisCTFSign = TImagesCTFFull[iterTrain].GetCopy();
                                thisCTFSign.Sign();
                                TImagesCTFFull[iterTrain].Multiply(thisCTFSign);
                                TImagesCTFFull[iterTrain].Multiply(CTFMaskFull);
                                thisCTFSign.Dispose();
                            }
                            {
                                Image thisCTFSign = TImagesCTFScaled[iterTrain].GetCopy();
                                thisCTFSign.Sign();
                                TImagesCTFScaled[iterTrain].Multiply(thisCTFSign);
                                TImagesCTFScaled[iterTrain].Multiply(CTFMaskScaled);
                                thisCTFSign.Dispose();
                            }
                            {
                                Image fft = projected.AsFFT();
                                projected.Dispose();
                                fft.Multiply(TImagesCTFFull[iterTrain]);
                                projected = fft.AsIFFT(false, 0, true);
                                fft.Dispose();
                            }

                            Image ColoredNoiseFFT = new Image(projected.Dims, true, true);
                            float2[][] complexData = Helper.ArrayOfFunction(i => new float2[projected.DimsFTSlice.Elements()], projected.Dims.Z);
                            for (int z = 0; z < projected.Dims.Z; z++)
                            {
                                for (int y = 0; y < projected.Dims.Y; y++)
                                {
                                    for (int x = 0; x < projected.Dims.X/2+1; x++)
                                    {
                                        float yy = y >= projected.Dims.Y / 2 + 1 ? y - projected.Dims.Y : y;
                                        yy *= yy;

                                        float xx = x;
                                        xx *= xx;

                                        float r = (float)Math.Sqrt(xx + yy);
                                        int waveNumber = (int)Math.Round(r);
                                        waveNumber = Math.Min(waveNumber, projected.Dims.X / 2);
                                        complexData[z][y*(projected.Dims.X / 2 + 1 )+ x] = new float2(NoiseRand.NextSingle(0, theseSigmas[z][waveNumber]), NoiseRand.NextSingle(0, theseSigmas[z][waveNumber]));
                                        
                                    }
                                }
                            }
                            ColoredNoiseFFT.UpdateHostWithComplex(complexData);
                            Image ColoredNoise = ColoredNoiseFFT.AsIFFT();
                            ColoredNoise.Normalize();
                            Image NoiseScaled = ColoredNoise.AsScaled(new int2(DimGenerator));
                            GPU.CopyDeviceToDevice(NoiseScaled.GetDevice(Intent.Read), TImagesNoise[iterTrain].GetDevice(Intent.Write), NoiseScaled.ElementsReal);
                            NoiseScaled.Dispose();
                            ColoredNoiseFFT.Dispose();
                            projected.Add(ColoredNoise);
                            ColoredNoise.Dispose();
                            /*
                            projected.TransformValues(val =>
                            {
                                //https://stackoverflow.com/a/218600/5012099
                                double u1 = 1.0 - NoiseRand.NextDouble(); //uniform(0,1] random doubles
                                double u2 = 1.0 - NoiseRand.NextDouble();
                                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                             Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
                                double randNormal = 0 + 1.0 * randStdNormal;
                                return (float)(val + randNormal);
                            });*/
                            Image projectedScaled = projected.AsScaled(new int2(DimGenerator));
                            projectedScaled.Bandpass(0, 1.0f, false, 0.05f);
                            projected.Dispose();
                            GPU.CopyDeviceToDevice(projectedScaled.GetDevice(Intent.Read), TImagesReal[iterTrain].GetDevice(Intent.Write), TImagesReal[iterTrain].ElementsReal);
                            projectedScaled.Dispose();
                        }

                        OwnBatchUsed = false;
                    }

                    _ = ReloadBlock.WaitOne();
                    // If previously pushed batch from any thread has already been processed
                    if (HasBeenProcessed)
                    {
                        for (int discIter = 0; discIter < DiscIters + 1; discIter++)
                        {
                            GPU.CopyDeviceToDevice(TImagesReal[discIter].GetDevice(Intent.Read), ImagesReal[discIter].GetDevice(Intent.Write), TImagesReal[discIter].ElementsReal);
                            GPU.CopyDeviceToDevice(TImagesNoise[discIter].GetDevice(Intent.Read), ImagesNoise[discIter].GetDevice(Intent.Write), ImagesNoise[discIter].ElementsReal);
                            GPU.CopyDeviceToDevice(TImagesCTFScaled[discIter].GetDevice(Intent.Read), ImagesCTF[discIter].GetDevice(Intent.Write), TImagesCTFScaled[discIter].ElementsReal);
                            TImagesAngles[discIter].CopyTo(ImagesAngles[discIter], 0);
                            GPU.CheckGPUExceptions();
                        }

                        OwnBatchUsed = true;
                        HasBeenProcessed = false;
                    }
                    _ = ReloadBlock.Release();
                }
            };

            CancellationTokenSource ThreadSource = new();
            Thread[] ReloadThreads = Helper.ArrayOfFunction(i => new Thread(ReloadLambda), NThreads);
            for (int i = 0; i < NThreads; i++)
            {
                ReloadThreads[i].Start(new ThreadArgs(ThreadSource.Token, i));
            }

            GPU.SetDevice(ProcessingDevice);

            Random Rand = new Random(123);

            List<ObservablePoint> LossPointsReal = new List<ObservablePoint>();
            List<ObservablePoint> LossPointsFake = new List<ObservablePoint>();
            List<ObservablePoint> GradNormDiscPoints = new List<ObservablePoint>();
            List<ObservablePoint> GradNormGenPoints = new List<ObservablePoint>();

            long IterationsDone = 0;

            while (true)
            {
                if (HasBeenProcessed)
                {
                    continue;
                }

                if ((IterationsDone + 1) * BatchSize < 2 * numParticles)
                {
                    double clipVal = (IterationsDone + 1) * BatchSize * (DiscIters + 1) * 1e8 / (2 * numParticles);
                    TrainModel.set_discriminator_grad_clip_val(clipVal);
                    clipVal = (IterationsDone + 1) * BatchSize * (DiscIters + 1) * 1e4 / (2 * numParticles);
                    TrainModel.set_generator_grad_clip_val(clipVal);
                }

                _ = ReloadBlock.WaitOne();

                List<float> AllLossesReal = new List<float>();
                List<float> AllLossesFake = new List<float>();
                List<float> AllGradNormDisc = new List<float>();
                List<float> AllGradNormGen = new List<float>();
                float[] Loss = null;
                float[] LossReal = null;
                float[] LossFake = null;
                double gradNormDisc = 0.0d;
                double gradNormGen = 0.0d;
                Image PredictionDisc = null;
                Image PredictionGen = null;
                Image PredictionGenNoisy = null;
                {

                    float CurrentLearningRate = 0;
                    _ = Dispatcher.Invoke(() => CurrentLearningRate = (float)LearningRate);

                    if (ShouldTrainOnlyGen)
                    {
                        for (int iterDisc = 0; iterDisc <= DiscIters; iterDisc++)
                        {
                            TrainModel.TrainGeneratorParticle(ImagesAngles[iterDisc], ImagesCTF[iterDisc], ImagesReal[iterDisc], ImagesNoise[iterDisc],
                              CurrentLearningRate,
                              out PredictionGen,
                              out PredictionGenNoisy,
                              out Loss,
                              out gradNormGen);
                            AllGradNormGen.Add((float)gradNormGen);
                        }

                        TrainModel.TrainDiscriminatorParticle(ImagesAngles[DiscIters],
                              ImagesReal[DiscIters],
                              ImagesNoise[DiscIters],
                              ImagesCTF[DiscIters],
                              CurrentLearningRate,
                              Lambda,
                              out PredictionDisc,
                              out Loss,
                              out LossReal,
                              out LossFake,
                              out gradNormDisc);
                        AllLossesReal.Add(LossReal[0]);
                        AllLossesFake.Add(LossFake[0]);
                        AllGradNormDisc.Add((float)gradNormDisc);

                    }
                    else
                    {
                        for (int iterDisc = 0; iterDisc < DiscIters; iterDisc++)
                        {

                            TrainModel.TrainDiscriminatorParticle(ImagesAngles[iterDisc],
                                                                  ImagesReal[iterDisc],
                                                                  ImagesNoise[iterDisc],
                                                                  ImagesCTF[iterDisc],
                                                                  CurrentLearningRate,
                                                                  Lambda,
                                                                  out PredictionDisc,
                                                                  out Loss,
                                                                  out LossReal,
                                                                  out LossFake,
                                                                  out gradNormDisc);
                            AllGradNormDisc.Add((float)gradNormDisc);
                        }
                        AllLossesReal.Add(LossReal[0]);
                        AllLossesFake.Add(LossFake[0]);

                        if (TrainGen)
                        {
                            TrainModel.TrainGeneratorParticle(ImagesAngles[DiscIters], ImagesCTF[DiscIters], ImagesReal[DiscIters], ImagesNoise[DiscIters],
                                                              CurrentLearningRate,
                                                              out PredictionGen,
                                                              out PredictionGenNoisy,
                                                              out Loss,
                                                              out gradNormGen);
                            AllGradNormGen.Add((float)gradNormGen);
                        }
                    }
                    HasBeenProcessed = true;
                }

                if (IterationsDone % 10 == 0)
                {

                    WriteToLog($"{MathHelper.Mean(AllLossesReal):#.##E+00}, {MathHelper.Mean(AllLossesFake):#.##E+00}, {MathHelper.Max(AllGradNormDisc):#.##E+00}, {MathHelper.Max(AllGradNormGen):#.##E+00}");

                    LossPointsReal.Add(new ObservablePoint(IterationsDone, -1 * MathHelper.Mean(AllLossesReal)));
                    Dispatcher.Invoke(() => SeriesLossReal.Values = new ChartValues<ObservablePoint>(LossPointsReal));

                    LossPointsFake.Add(new ObservablePoint(IterationsDone, MathHelper.Mean(AllLossesFake)));
                    Dispatcher.Invoke(() => SeriesLossFake.Values = new ChartValues<ObservablePoint>(LossPointsFake));

                    GradNormDiscPoints.Add(new ObservablePoint(IterationsDone, MathHelper.Max(AllGradNormDisc)));
                    Dispatcher.Invoke(() => SeriesGradNorm.Values = new ChartValues<ObservablePoint>(GradNormDiscPoints));

                    GradNormGenPoints.Add(new ObservablePoint(IterationsDone, MathHelper.Max(AllGradNormGen)));
                    Dispatcher.Invoke(() => SeriesGradNormGen.Values = new ChartValues<ObservablePoint>(GradNormGenPoints));


                    Func<float[], float2, ImageSource> MakeImage = (data, stat) =>
                    {
                        float[] OneSlice = data.ToArray();

                        byte[] BytesXY = new byte[OneSlice.Length];
                        for (int y = 0; y < DimGenerator; y++)
                        {
                            for (int x = 0; x < DimGenerator; x++)
                            {
                                float Value = (OneSlice[y * DimGenerator + x] - stat.X) / stat.Y;
                                Value = (Value + 4f) / 8f;
                                BytesXY[(DimGenerator - 1 - y) * DimGenerator + x] = (byte)(Math.Max(0, Math.Min(1, Value)) * 255);
                            }
                        }

                        ImageSource SliceImage = BitmapSource.Create(DimGenerator, DimGenerator, 96, 96, PixelFormats.Indexed8, BitmapPalettes.Gray256, BytesXY, DimGenerator);
                        SliceImage.Freeze();

                        return SliceImage;
                    };

                    float[] all_data = new List<float>().Concat(ImagesReal[DiscIters].GetHost(Intent.Read)[0]).Concat(PredictionGenNoisy.GetHost(Intent.Read)[0]).Concat(PredictionGen.GetHost(Intent.Read)[0]).ToArray();
                    float2 stat = MathHelper.MeanAndStd(all_data);
                    {
                        ImageSource SliceImage = MakeImage(ImagesReal[DiscIters].GetHost(Intent.Read)[0], stat);
                        _ = Dispatcher.Invoke(() => ImageSource.Source = SliceImage);
                    }

                    {
                        ImageSource SliceImage = MakeImage(PredictionGenNoisy.GetHost(Intent.Read)[0], stat);
                        _ = Dispatcher.Invoke(() => ImageTarget.Source = SliceImage);
                    }

                    {
                        ImageSource SliceImage = MakeImage(PredictionGen.GetHost(Intent.Read)[0], stat);
                        _ = Dispatcher.Invoke(() => ImageAverage.Source = SliceImage);
                    }

                    AllLossesReal.Clear();
                    AllLossesFake.Clear();
                }

                IterationsDone++;
                _ = Dispatcher.Invoke(() => TextCoverage.Text = $"{IterationsDone} iterations done");

                if ((IterationsDone * BatchSize * (DiscIters + 1)) > (currentEpoch + 1) * numParticles)
                {
                    if (currentEpoch %10 ==0 || currentEpoch==numEpochs-1)
                    {
                        ShouldSaveModel = true;
                    }
                    currentEpoch += 1;
                    _ = Dispatcher.Invoke(() => LearningRate = LearningRate * reduction);                   
                }

                if (ShouldSaveModel)
                {
                    ShouldSaveModel = false;
                    string datestring = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                    TrainModel.Save(OutDirectory + @"ParticleWGAN_SN_" + datestring + ".pt");
                    Image imageVolume = TrainModel.getVolume();
                    imageVolume.WriteMRC(OutDirectory + @"ParticleWGAN_SN_" + datestring + ".mrc", true);
                    Image resized;
                    if (DimGenerator != DimZoom)
                        resized = imageVolume.AsScaled(new int3(DimZoom));
                    else
                        resized = imageVolume;

                    if (DimZoom != DimRaw)
                        resized = resized.AsPadded(new int3(DimRaw));
                    resized.WriteMRC(OutDirectory + @"ParticleWGAN_SN_" + datestring + "_resized.mrc", true);
                    resized.MaskSpherically(DimGenerator / 2, DimGenerator / 8, false);
                    resized.WriteMRC(OutDirectory + @"ParticleWGAN_SN_" + datestring + "_resized_masked.mrc", true);
                    imageVolume.Dispose();
                    Thread.Sleep(10000);

                    Dispatcher.Invoke(() => ButtonSave.IsEnabled = true);

                    if (ShouldSaveRecs)
                    {
                        Image.Stack(ImagesReal).WriteMRC(OutDirectory + @"ParticleWGAN_SN_" + datestring + "_real.mrc", true);
                        Image.Stack(ImagesCTF).WriteMRC(OutDirectory + @"ParticleWGAN_SN_" + datestring + "_ctf.mrc", true);
                        PredictionGen.WriteMRC(OutDirectory + @"ParticleWGAN_SN_" + datestring + "_prediction.mrc", true);
                        PredictionGenNoisy.WriteMRC(OutDirectory + @"ParticleWGAN_SN_" + datestring + "_predictionNoisy.mrc", true);
                    }

                }
                if (currentEpoch >= numEpochs)
                {
                    //Tell Threads to exit
                    ThreadSource.Cancel();
                    //Release wait block, so threads can actually exit
                    ReloadBlock.Release();
                    for (int i = 0; i < NThreads; i++)
                    {
                        //Wait for all threads
                        ReloadThreads[i].Join();
                    }
                    //kill this application
                    Application.Current.Dispatcher.Invoke(() => this.CloseMainwindow());
                    return;
                }
                _ = ReloadBlock.Release();
            }
        }

        protected override void OnClosed(EventArgs e)
        {
            base.OnClosed(e);

            Application.Current.Shutdown();
        }
        private void ButtonStartParticle_OnClick(object sender, RoutedEventArgs e)
        {
            ButtonStartParticle.IsEnabled = false;
            doTraining();
            Task.Run(doTraining);
        }

        private void ButtonSave_OnClick(object sender, RoutedEventArgs e)
        {
            ShouldSaveModel = true;
            ButtonSave.IsEnabled = false;
        }

        private void WriteToLog(string line)
        {
            Dispatcher.Invoke(() =>
            {
                TextOutput.Text += line + "\n";
                TextOutput.ScrollToLine(TextOutput.LineCount - 1);
            });
            if(logFileName.Length > 0)
            {
                using (StreamWriter w = File.AppendText(logFileName))
                {
                    w.WriteLine(line);
                }
            }
        }
        /*
        private void ButtonTest_OnClick_old(object sender, RoutedEventArgs e)
        {
            WriteToLog("Loading model... (" + GPU.GetFreeMemory(0) + " MB free)");
            ParticleWGAN TrainModel = new ParticleWGAN(new int2(DimGenerator), 32, new[] { 0 }, BatchSize);
            TrainModel.Load(WorkingDirectory + "ParticleWGAN_20210108_002634.pt");
            WriteToLog("Done. (" + GPU.GetFreeMemory(0) + " MB free)");

            GPU.SetDevice(PreProcessingDevice);

            Image[] ImagesReal = null;
            Image[] ImagesFake = null;
            Image[] ImagesCTF = null;

            Semaphore ReloadBlock = new Semaphore(1, 1);
            bool HasBeenProcessed = true;

            Star TableIn = new Star(@"E:\particleWGAN\c4_coords.star", "particles");

            string[] ColumnStackNames = TableIn.GetColumn("rlnImageName").Select(s => Helper.PathToNameWithExtension(s.Substring(s.IndexOf('@') + 1))).ToArray();
            HashSet<string> UniqueStackNames = Helper.GetUniqueElements(ColumnStackNames);
            UniqueStackNames.RemoveWhere(s => !File.Exists(Path.Combine(@"E:\particleWGAN\real", s)));
            int[] KeepRows = Helper.ArrayOfSequence(0, TableIn.RowCount, 1).Where(r => UniqueStackNames.Contains(ColumnStackNames[r])).ToArray();
            TableIn = TableIn.CreateSubset(KeepRows);

            int DimRaw = MapHeader.ReadFromFile(Path.Combine(WorkingDirectory, DirectoryReal, UniqueStackNames.First())).Dimensions.X;

            TableIn.AddColumn("rlnVoltage", "200.0");
            TableIn.AddColumn("rlnSphericalAberration", "2.7");
            TableIn.AddColumn("rlnAmplitudeContrast", "0.07");
            TableIn.AddColumn("rlnDetectorPixelSize", "1.5");
            TableIn.AddColumn("rlnMagnification", "10000");

            var AllParticleAddresses = new (int id, string name)[TableIn.RowCount];
            {
                ColumnStackNames = TableIn.GetColumn("rlnImageName");
                for (int r = 0; r < TableIn.RowCount; r++)
                {
                    string s = ColumnStackNames[r];
                    int ID = int.Parse(s.Substring(0, s.IndexOf('@'))) - 1;
                    string Name = Helper.PathToNameWithExtension(s.Substring(s.IndexOf('@') + 1));
                    AllParticleAddresses[r] = (ID, Name);
                }
            }
            CTF[] AllParticleCTF = TableIn.GetRelionCTF();
            int[] AllIDs = Helper.ArrayOfSequence(0, AllParticleAddresses.Length, 1);

            ParameterizedThreadStart ReloadLambda = (par) =>
            {
                GPU.SetDevice(PreProcessingDevice);

                Random ReloadRand = new Random((int)par);
                bool OwnBatchUsed = false;

                Image LoadStack = new Image(new int3(DimRaw, DimRaw, BatchSize));

                Image TImagesFake = new Image(new int3(DimGenerator, DimGenerator, BatchSize));
                Image TImagesCTF = new Image(new int3(DimGenerator, DimGenerator, BatchSize), true);

                Image CTFCoords = CTF.GetCTFCoords(DimGenerator, DimRaw);

                int PlanForw = 0, PlanBack = 0;
                if (DimRaw != DimGenerator)
                {
                    PlanForw = GPU.CreateFFTPlan(new int3(DimRaw, DimRaw, 1), (uint)BatchSize);
                    PlanBack = GPU.CreateIFFTPlan(new int3(DimGenerator, DimGenerator, 1), (uint)BatchSize);
                }

                int NDone = 0;

                while (true)
                {
                    // If this thread succeeded at pushing its previously loaded batch to processing
                    int[] SubsetIDs = Helper.ArrayOfConstant(0, BatchSize);

                    float[][] LoadStackData = LoadStack.GetHost(Intent.Write);

                    for (int b = 0; b < BatchSize; b++)
                    {
                        int id = SubsetIDs[b];
                        IOHelper.ReadMapFloat(Path.Combine(WorkingDirectory, DirectoryFake, AllParticleAddresses[id].name),
                                                new int2(1),
                                                0,
                                                typeof(float),
                                                new[] { AllParticleAddresses[id].id },
                                                null,
                                                new[] { LoadStackData[b] });
                    }

                    if (DimGenerator == DimRaw)
                        GPU.CopyDeviceToDevice(LoadStack.GetDevice(Intent.Read),
                                                TImagesFake.GetDevice(Intent.Write),
                                                LoadStack.ElementsReal);
                    else
                        GPU.Scale(LoadStack.GetDevice(Intent.Read),
                                    TImagesFake.GetDevice(Intent.Write),
                                    LoadStack.Dims.Slice(),
                                    TImagesFake.Dims.Slice(),
                                    (uint)BatchSize,
                                    PlanForw,
                                    PlanBack,
                                    IntPtr.Zero,
                                    IntPtr.Zero);

                    GPU.CheckGPUExceptions();

                    GPU.CreateCTF(TImagesCTF.GetDevice(Intent.Write),
                                    CTFCoords.GetDevice(Intent.Read),
                                    IntPtr.Zero,
                                    (uint)CTFCoords.ElementsSliceComplex,
                                    Helper.IndexedSubset(AllParticleCTF, SubsetIDs).Select(c => c.ToStruct()).ToArray(),
                                    false,
                                    (uint)BatchSize);

                    Image Predicted = null;

                    TrainModel.Predict(TImagesFake, TImagesCTF, out Predicted);

                    Predicted.WriteMRC(Path.Combine(WorkingDirectory, "ganned", $"{NDone++}.mrc"), true);

                    if (NDone == 100)
                        break;
                }
            };
            Thread[] ReloadThreads = Helper.ArrayOfFunction(i => new Thread(ReloadLambda), NThreads);
            for (int i = 0; i < NThreads; i++)
                ReloadThreads[i].Start(i);
        }

        private void ButtonTest_OnClick(object sender, RoutedEventArgs e)
        {
            NThreads = 1;

            WriteToLog("Loading model... (" + GPU.GetFreeMemory(0) + " MB free)");
            ParticleWGAN TrainModel = new ParticleWGAN(new int2(DimGenerator), 64, new[] { 1 }, BatchSize);
            TrainModel.Load(WorkingDirectory + "ParticleWGAN_20210111_210604.pt");
            WriteToLog("Done. (" + GPU.GetFreeMemory(0) + " MB free)");

            GPU.SetDevice(PreProcessingDevice);

            Image[] ImagesReal = null;
            Image[] ImagesFake = null;
            Image[] ImagesCTF = null;

            Semaphore ReloadBlock = new Semaphore(1, 1);
            bool HasBeenProcessed = true;

            Star TableIn = new Star(Path.Combine(WorkingDirectory, "c4_coords.star"), "particles");

            string[] ColumnStackNames = TableIn.GetColumn("rlnImageName").Select(s => Helper.PathToNameWithExtension(s.Substring(s.IndexOf('@') + 1))).ToArray();
            HashSet<string> UniqueStackNames = Helper.GetUniqueElements(ColumnStackNames);
            UniqueStackNames.RemoveWhere(s => !File.Exists(Path.Combine(WorkingDirectory, DirectoryReal, s)));
            int[] KeepRows = Helper.ArrayOfSequence(0, TableIn.RowCount, 1).Where(r => UniqueStackNames.Contains(ColumnStackNames[r])).ToArray();
            TableIn = TableIn.CreateSubset(KeepRows);

            int DimRaw = MapHeader.ReadFromFile(Path.Combine(WorkingDirectory, DirectoryReal, UniqueStackNames.First())).Dimensions.X;

            TableIn.AddColumn("rlnVoltage", "200.0");
            TableIn.AddColumn("rlnSphericalAberration", "2.7");
            TableIn.AddColumn("rlnAmplitudeContrast", "0.07");
            TableIn.AddColumn("rlnDetectorPixelSize", "1.5");
            TableIn.AddColumn("rlnMagnification", "10000");

            var AllParticleAddresses = new (int id, string name)[TableIn.RowCount];
            {
                ColumnStackNames = TableIn.GetColumn("rlnImageName");
                for (int r = 0; r < TableIn.RowCount; r++)
                {
                    string s = ColumnStackNames[r];
                    int ID = int.Parse(s.Substring(0, s.IndexOf('@'))) - 1;
                    string Name = Helper.PathToNameWithExtension(s.Substring(s.IndexOf('@') + 1));
                    AllParticleAddresses[r] = (ID, Name);
                }
            }
            CTF[] AllParticleCTF = TableIn.GetRelionCTF();
            int[] AllIDs = Helper.ArrayOfSequence(0, AllParticleAddresses.Length, 1);

            {
                Image LoadStack = new Image(new int3(DimRaw, DimRaw, BatchSize));

                Image TImagesFake = new Image(new int3(DimGenerator, DimGenerator, BatchSize));
                Image TImagesCTF = new Image(new int3(DimGenerator, DimGenerator, BatchSize), true);

                Image CTFCoords = CTF.GetCTFCoords(DimGenerator, DimRaw);

                int PlanForw = 0, PlanBack = 0;
                if (DimRaw != DimGenerator)
                {
                    PlanForw = GPU.CreateFFTPlan(new int3(DimRaw, DimRaw, 1), (uint)BatchSize);
                    PlanBack = GPU.CreateIFFTPlan(new int3(DimGenerator, DimGenerator, 1), (uint)BatchSize);
                }

                List<Star> AllTables = new List<Star>();

                foreach (var stackName in UniqueStackNames)
                {
                    int[] StackIDs = Helper.ArrayOfSequence(0, AllParticleAddresses.Length, 1).Where(i => AllParticleAddresses[i].name == stackName).ToArray();
                    int NParticles = StackIDs.Length;

                    Image ParticleStack = new Image(new int3(DimGenerator - 16, DimGenerator - 16, NParticles));

                    for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                    {
                        int CurBatch = Math.Min(BatchSize, NParticles - batchStart);

                        // If this thread succeeded at pushing its previously loaded batch to processing
                        int[] SubsetIDs = Helper.Combine(StackIDs.Skip(batchStart).Take(CurBatch).ToArray(), Helper.ArrayOfConstant(StackIDs.Last(), BatchSize - CurBatch));

                        float[][] LoadStackData = LoadStack.GetHost(Intent.Write);

                        for (int b = 0; b < BatchSize; b++)
                        {
                            int id = SubsetIDs[b];
                            IOHelper.ReadMapFloat(Path.Combine(WorkingDirectory, DirectoryFake, AllParticleAddresses[id].name),
                                                    new int2(1),
                                                    0,
                                                    typeof(float),
                                                    new[] { AllParticleAddresses[id].id },
                                                    null,
                                                    new[] { LoadStackData[b] });
                        }

                        if (DimGenerator == DimRaw)
                            GPU.CopyDeviceToDevice(LoadStack.GetDevice(Intent.Read),
                                                    TImagesFake.GetDevice(Intent.Write),
                                                    LoadStack.ElementsReal);
                        else
                            GPU.Scale(LoadStack.GetDevice(Intent.Read),
                                        TImagesFake.GetDevice(Intent.Write),
                                        LoadStack.Dims.Slice(),
                                        TImagesFake.Dims.Slice(),
                                        (uint)BatchSize,
                                        PlanForw,
                                        PlanBack,
                                        IntPtr.Zero,
                                        IntPtr.Zero);

                        GPU.CheckGPUExceptions();

                        GPU.CreateCTF(TImagesCTF.GetDevice(Intent.Write),
                                        CTFCoords.GetDevice(Intent.Read),
                                        IntPtr.Zero,
                                        (uint)CTFCoords.ElementsSliceComplex,
                                        Helper.IndexedSubset(AllParticleCTF, SubsetIDs).Select(c => c.ToStruct()).ToArray(),
                                        false,
                                        (uint)BatchSize);

                        Image Predicted = null;

                        //TImagesFake.WriteMRC(Path.Combine(WorkingDirectory, $"ref.mrc"), true);

                        TrainModel.Predict(TImagesFake, TImagesCTF, out Predicted);

                        //Predicted.WriteMRC(Path.Combine(WorkingDirectory, "ganned", $"{NDone++}_crap.mrc"), true);

                        GPU.Pad(Predicted.GetDevice(Intent.Read),
                                ParticleStack.GetDeviceSlice(batchStart, Intent.Write),
                                new int3(DimGenerator, DimGenerator, 1),
                                ParticleStack.Dims.Slice(),
                                (uint)CurBatch);
                    }

                    ParticleStack.SubtractMeanGrid(new int2(1));
                    GPU.NormParticles(ParticleStack.GetDevice(Intent.Read),
                                      ParticleStack.GetDevice(Intent.Write),
                                      ParticleStack.Dims.Slice(),
                                      (uint)(DimGenerator / 4),
                                      false,
                                      (uint)NParticles);

                    ParticleStack.WriteMRC(Path.Combine(WorkingDirectory, "ganparticles_std", stackName), true);
                    ParticleStack.Dispose();

                    Star StackTable = TableIn.CreateSubset(StackIDs);
                    for (int r = 0; r < NParticles; r++)
                    {
                        StackTable.SetRowValue(r, "rlnImageName", $"{(r + 1):D5}@{stackName}");
                    }

                    StackTable.SetColumn("rlnDetectorPixelSize", Helper.ArrayOfConstant("3.0", NParticles));
                    AllTables.Add(StackTable);
                }

                new Star(AllTables.ToArray()).Save(Path.Combine(WorkingDirectory, "ganparticles_std", "particles.star"));
            }
        }
        */
    }
}
