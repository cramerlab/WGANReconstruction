using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommandLine;
using CommandLine.Text;
using Warp;
using Warp.Tools;

namespace Noise2Map
{
    class Program
    {
        static void Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

            #region Command line options

            Options Options = new Options();
            string WorkingDirectory;

            string ProgramFolder = System.Reflection.Assembly.GetEntryAssembly().Location;
            ProgramFolder = ProgramFolder.Substring(0, Math.Max(ProgramFolder.LastIndexOf('\\'), ProgramFolder.LastIndexOf('/')) + 1);

            if (!Debugger.IsAttached)
            {
                Parser.Default.ParseArguments<Options>(args).WithParsed<Options>(opts => Options = opts);
                WorkingDirectory = Environment.CurrentDirectory + "/";
            }
            else
            {
                Options.Observation1Path = @"X:\dtegunov\corona_insitu\1907session\reconstruction\odd";
                Options.Observation2Path = @"X:\dtegunov\corona_insitu\1907session\reconstruction\even";
                Options.ObservationCombinedPath = @"";
                Options.CTFPath = @"X:\dtegunov\corona_insitu\1907session\reconstruction\ctf";
                Options.DenoiseSeparately = false;
                Options.MaskPath = @"";
                Options.DontAugment = true;
                Options.DontFlatten = true;
                Options.Overflatten = 1.0f;
                Options.PixelSize = 15f;
                Options.Upsample = 1.0f;
                Options.Lowpass = -1;
                Options.KeepDimensions = true;
                Options.MaskOutput = false;
                Options.NIterations = 600;
                Options.BatchSize = 4;
                Options.GPUNetwork = 0;
                Options.GPUPreprocess = 2;
                Options.OldModelName = "";
                WorkingDirectory = @"X:\dtegunov\corona_insitu\1907session\reconstruction\";
            }

            int3 TrainingDims = new int3(192, 64, 192);

            if (!Options.DontFlatten && Options.PixelSize < 0)
                throw new Exception("Flattening requested, but pixel size not specified.");

            if (!Options.DontAugment && !string.IsNullOrEmpty(Options.CTFPath))
                throw new Exception("3D CTF cannot be combined with data augmentation.");

            #endregion

            int NDevices = GPU.GetDeviceCount();
            if (Options.GPUNetwork >= NDevices || Options.GPUPreprocess >= NDevices)
            {
                Console.WriteLine("Requested GPU ID that isn't present on this system. Defaulting to highest ID available.\n");

                Options.GPUNetwork = Math.Min(Options.GPUNetwork, NDevices - 1);
                Options.GPUPreprocess = Math.Min(Options.GPUPreprocess, NDevices - 1);
            }

            GPU.SetDevice(Options.GPUPreprocess);

            #region Mask

            Console.Write("Loading mask... ");

            Image Mask = null;
            int3 BoundingBox = new int3(-1);
            if (!string.IsNullOrEmpty(Options.MaskPath))
            {
                Mask = Image.FromFile(Options.MaskPath);

                Mask.TransformValues((x, y, z, v) =>
                {
                    if (v > 1e-3f)
                    {
                        BoundingBox.X = Math.Max(BoundingBox.X, Math.Abs(x - Mask.Dims.X / 2) * 2);
                        BoundingBox.Y = Math.Max(BoundingBox.Y, Math.Abs(y - Mask.Dims.Y / 2) * 2);
                        BoundingBox.Z = Math.Max(BoundingBox.Z, Math.Abs(z - Mask.Dims.Z / 2) * 2);
                    }

                    return v;
                });

                if (BoundingBox.X < 2)
                    throw new Exception("Mask does not seem to contain any non-zero values.");

                BoundingBox += 64;

                BoundingBox.X = Math.Min(BoundingBox.X, Mask.Dims.X);
                BoundingBox.Y = Math.Min(BoundingBox.Y, Mask.Dims.Y);
                BoundingBox.Z = Math.Min(BoundingBox.Z, Mask.Dims.Z);
            }

            Console.WriteLine("done.\n");

            #endregion

            #region Load and prepare data

            Console.WriteLine("Preparing data:");

            List<Image> Maps1 = new List<Image>();
            List<Image> Maps2 = new List<Image>();
            List<Image> MapCTFs = new List<Image>();
            List<ulong[]> Textures1 = new List<ulong[]>();
            List<ulong[]> Textures2 = new List<ulong[]>();
            List<Image> MapsForDenoising = new List<Image>();
            List<Image> MapsForDenoising2 = new List<Image>();
            List<string> NamesForDenoising = new List<string>();
            List<int3> DimensionsForDenoising = new List<int3>();
            List<int3> OriginalBoxForDenoising = new List<int3>();
            List<float2> MeanStdForDenoising = new List<float2>();
            List<float> PixelSizeForDenoising = new List<float>();

            foreach (var file in Directory.EnumerateFiles(Options.Observation1Path, "*.mrc"))
            {
                string MapName = Helper.PathToName(file);
                string[] Map2Paths = Directory.EnumerateFiles(Options.Observation2Path, MapName + ".mrc").ToArray();
                if (Map2Paths == null || Map2Paths.Length == 0)
                    continue;
                string MapCombinedPath = null;
                if (!string.IsNullOrEmpty(Options.ObservationCombinedPath))
                {
                    string[] MapCombinedPaths = Directory.EnumerateFiles(Options.ObservationCombinedPath, MapName + ".mrc").ToArray();
                    if (MapCombinedPaths == null || MapCombinedPaths.Length == 0)
                        continue;
                    MapCombinedPath = MapCombinedPaths.First();
                }

                Console.Write($"Preparing {MapName}... ");

                Image Map1 = Image.FromFile(file);
                Image Map2 = Image.FromFile(Map2Paths.First());
                Image MapCombined = MapCombinedPath == null ? null : Image.FromFile(MapCombinedPath);

                float MapPixelSize = Map1.PixelSize / (Options.KeepDimensions ? 1 : Options.Upsample);

                if (!Options.DontFlatten)
                {
                    Image Average = Map1.GetCopy();
                    Average.Add(Map2);

                    if (Mask != null)
                        Average.Multiply(Mask);

                    float[] Spectrum = Average.AsAmplitudes1D(true, 1, (Average.Dims.X + Average.Dims.Y + Average.Dims.Z) / 6);
                    Average.Dispose();

                    int i10A = (int)(Options.PixelSize * 2 / 10 * Spectrum.Length);
                    float Amp10A = Spectrum[i10A];

                    for (int i = 0; i < Spectrum.Length; i++)
                        Spectrum[i] = i < i10A ? 1 : (float)Math.Pow(Amp10A / Spectrum[i], Options.Overflatten);

                    Image Map1Flat = Map1.AsSpectrumMultiplied(true, Spectrum);
                    Map1.Dispose();
                    Map1 = Map1Flat;
                    Map1.FreeDevice();

                    Image Map2Flat = Map2.AsSpectrumMultiplied(true, Spectrum);
                    Map2.Dispose();
                    Map2 = Map2Flat;
                    Map2.FreeDevice();

                    if (MapCombined != null)
                    {
                        Image MapCombinedFlat = MapCombined.AsSpectrumMultiplied(true, Spectrum);
                        MapCombined.Dispose();
                        MapCombined = MapCombinedFlat;
                        MapCombined.FreeDevice();
                    }
                }

                if (Options.Lowpass > 0)
                {
                    Map1.Bandpass(0, Options.PixelSize * 2 / Options.Lowpass, true, 0.01f);
                    Map2.Bandpass(0, Options.PixelSize * 2 / Options.Lowpass, true, 0.01f);
                    MapCombined?.Bandpass(0, Options.PixelSize * 2 / Options.Lowpass, true, 0.01f);
                }

                OriginalBoxForDenoising.Add(Map1.Dims);

                if (BoundingBox.X > 0)
                {
                    Image Map1Cropped = Map1.AsPadded(BoundingBox);
                    Map1.Dispose();
                    Map1 = Map1Cropped;
                    Map1.FreeDevice();

                    Image Map2Cropped = Map2.AsPadded(BoundingBox);
                    Map2.Dispose();
                    Map2 = Map2Cropped;
                    Map2.FreeDevice();

                    if (MapCombined != null)
                    {
                        Image MapCombinedCropped = MapCombined.AsPadded(BoundingBox);
                        MapCombined.Dispose();
                        MapCombined = MapCombinedCropped;
                        MapCombined.FreeDevice();
                    }
                }

                DimensionsForDenoising.Add(Map1.Dims);

                if (Options.Upsample != 1f)
                {
                    Image Map1Scaled = Map1.AsScaled(Map1.Dims * Options.Upsample / 2 * 2);
                    Map1.Dispose();
                    Map1 = Map1Scaled;
                    Map1.FreeDevice();

                    Image Map2Scaled = Map2.AsScaled(Map2.Dims * Options.Upsample / 2 * 2);
                    Map2.Dispose();
                    Map2 = Map2Scaled;
                    Map2.FreeDevice();

                    if (MapCombined != null)
                    {
                        Image MapCombinedScaled = MapCombined.AsScaled(Map2.Dims * Options.Upsample / 2 * 2);
                        MapCombined.Dispose();
                        MapCombined = MapCombinedScaled;
                        MapCombined.FreeDevice();
                    }
                }

                float2 MeanStd = MathHelper.MeanAndStd(Helper.Combine(Map1.GetHostContinuousCopy(), Map2.GetHostContinuousCopy()));
                MeanStdForDenoising.Add(MeanStd);

                Map1.TransformValues(v => (v - MeanStd.X) / MeanStd.Y);
                Map2.TransformValues(v => (v - MeanStd.X) / MeanStd.Y);
                MapCombined?.TransformValues(v => (v - MeanStd.X) / MeanStd.Y);

                Image ForDenoising = (MapCombined == null || Options.DenoiseSeparately) ? Map1.GetCopy() : MapCombined;
                Image ForDenoising2 = Options.DenoiseSeparately ? Map2.GetCopy() : null;

                GPU.PrefilterForCubic(Map1.GetDevice(Intent.ReadWrite), Map1.Dims);
                Map1.FreeDevice();
                Maps1.Add(Map1);

                if (!Options.DenoiseSeparately)
                {
                    ForDenoising.Add(Map2);
                    ForDenoising.Multiply(0.5f);
                }

                GPU.PrefilterForCubic(Map2.GetDevice(Intent.ReadWrite), Map2.Dims);
                Map2.FreeDevice();
                Maps2.Add(Map2);

                ForDenoising.FreeDevice();
                MapsForDenoising.Add(ForDenoising);
                NamesForDenoising.Add(MapName);

                PixelSizeForDenoising.Add(MapPixelSize);

                if (Options.DenoiseSeparately)
                {
                    ForDenoising2.FreeDevice();
                    MapsForDenoising2.Add(ForDenoising2);
                }

                if (!string.IsNullOrEmpty(Options.CTFPath) &&
                    File.Exists(Path.Combine(WorkingDirectory, Options.CTFPath, MapName + ".mrc")))
                {
                    Image MapCTF = Image.FromFile(Path.Combine(WorkingDirectory, Options.CTFPath, MapName + ".mrc"));
                    {
                        MapCTF.Dims = new int3(64, 64, 64);
                        MapCTF.IsFT = true;
                        Image CTFComplex = new Image(MapCTF.Dims, true, true);
                        CTFComplex.Fill(new float2(1, 0));
                        CTFComplex.Multiply(MapCTF);
                        MapCTF.Dispose();
                        Image CTFReal = CTFComplex.AsIFFT(true).AndDisposeParent();
                        Image CTFPadded = CTFReal.AsPadded(TrainingDims * 2, true).AndDisposeParent();
                        CTFComplex = CTFPadded.AsFFT(true).AndDisposeParent();
                        MapCTF = CTFComplex.AsReal().AndDisposeParent();
                        MapCTF.Multiply(1f / (64 * 64 * 64));

                        MapCTF.WriteMRC(Path.Combine(WorkingDirectory, Options.CTFPath, MapName + "_scaled.mrc"), true);
                    }
                    float[] CTFData = MapCTF.GetHost(Intent.ReadWrite)[0];
                    Helper.ForEachElementFT(TrainingDims * 2, (x, y, z, xx, yy, zz, r) =>
                    {
                        float xxx = xx / TrainingDims.X;
                        float yyy = yy / TrainingDims.Y;
                        float zzz = zz / TrainingDims.Z;

                        r = (float)Math.Sqrt(xxx * xxx + yyy * yyy + zzz * zzz);

                        r = Math.Min(1, r * 10);
                        r = (float)Math.Cos(r * Math.PI) * 0.5f + 0.5f;

                        int i = y * (MapCTF.Dims.X / 2 + 1) + x;
                        CTFData[i] = MathHelper.Lerp(CTFData[i], 1, r);
                    });
                    MapCTFs.Add(MapCTF);
                    Console.WriteLine("Found CTF");
                }
                else
                {
                    Image MapCTF = new Image(new int3(128), true);
                    MapCTF.TransformValues(v => 1f);
                    MapCTFs.Add(MapCTF);
                }

                Console.WriteLine($" Done.");// {GPU.GetFreeMemory(GPU.GetDevice())} MB");
                GPU.CheckGPUExceptions();
            }

            Mask?.FreeDevice();

            if (Maps1.Count == 0)
                throw new Exception("No maps were found. Please make sure the paths are correct and the names are consistent between the two observations.");

            Console.WriteLine("");

            #endregion

            NoiseNet3DTorch TrainModel = null;
            string NameTrainedModel = Options.OldModelName;
            int3 Dim = TrainingDims;
            int3 Dim2 = Dim * 2;

            if (Options.BatchSize != 4 || Maps1.Count > 1)
            {
                if (Options.BatchSize < 1)
                    throw new Exception("Batch size must be at least 1.");

                Options.NIterations = Options.NIterations * 4 / Options.BatchSize / Math.Min(8, Maps1.Count);
                Console.WriteLine($"Adjusting the number of iterations to {Options.NIterations} to match batch size and number of maps.\n");
            }


            if (string.IsNullOrEmpty(Options.OldModelName))
            {
                #region Load model

                string ModelPath = Options.StartModelName;
                if (!string.IsNullOrEmpty(ModelPath))
                {
                    if (!File.Exists(ModelPath))
                        ModelPath = Path.Combine(ProgramFolder, Options.StartModelName);
                    if (!File.Exists(ModelPath))
                        throw new Exception($"Could not find initial model '{Options.StartModelName}'. Please make sure it can be found either here, or in the installation directory.");
                }

                Console.WriteLine("Loading model, " + GPU.GetFreeMemory(Options.GPUNetwork) + " MB free.");
                TrainModel = new NoiseNet3DTorch(Dim, new[] { 0, 1, 2, 3 }, Options.BatchSize);
                if (!string.IsNullOrEmpty(ModelPath))
                    TrainModel.Load(ModelPath);
                Console.WriteLine("Loaded model, " + GPU.GetFreeMemory(Options.GPUNetwork) + " MB remaining.\n");

                #endregion

                GPU.SetDevice(Options.GPUPreprocess);

                #region Training

                Random Rand = new Random(123);

                int NMaps = Maps1.Count;
                int NMapsPerBatch = Math.Min(8, NMaps);
                int MapSamples = Options.BatchSize;

                Image[] ExtractedSource = Helper.ArrayOfFunction(i => new Image(new int3(Dim.X, Dim.Y, Dim.Z * MapSamples)), NMapsPerBatch);
                Image[] ExtractedSourceRand = Helper.ArrayOfFunction(i => new Image(new int3(Dim.X, Dim.Y, Dim.Z * MapSamples)), NMapsPerBatch);
                Image[] ExtractedTarget = Helper.ArrayOfFunction(i => new Image(new int3(Dim.X, Dim.Y, Dim.Z * MapSamples)), NMapsPerBatch);
                Image[] ExtractedTargetRand = Helper.ArrayOfFunction(i => new Image(new int3(Dim.X, Dim.Y, Dim.Z * MapSamples)), NMapsPerBatch);
                Image[] ExtractedCTF = Helper.ArrayOfFunction(i => new Image(new int3(Dim2.X, Dim2.Y, Dim2.Z * MapSamples), true), NMapsPerBatch);
                Image[] ExtractedCTFRand = Helper.ArrayOfFunction(i => new Image(new int3(Dim2.X, Dim2.Y, Dim2.Z * MapSamples), true), NMapsPerBatch);

                foreach (var item in MapCTFs)
                    item.GetDevice(Intent.Read);

                Stopwatch Watch = new Stopwatch();
                Watch.Start();

                Queue<float> Losses = new Queue<float>();

                for (int iter = 0; iter < Options.NIterations; iter++)
                {
                    int[] ShuffledMapIDs = Helper.RandomSubset(Helper.ArrayOfSequence(0, NMaps, 1), NMapsPerBatch, Rand.Next(9999999));

                    for (int m = 0; m < NMapsPerBatch; m++)
                    {
                        int MapID = ShuffledMapIDs[m];

                        Image Map1 = Maps1[MapID];
                        Image Map2 = Maps2[MapID];
                        //ulong Texture1 = Textures1[MapID][0];
                        //ulong Texture2 = Textures2[MapID][0];

                        int3 DimsMap = Map1.Dims;

                        int3 Margin = Dim / 2;
                        //Margin.Z = 0;
                        float3[] Position = Helper.ArrayOfFunction(i => new float3((float)Rand.NextDouble() * (DimsMap.X - Margin.X * 2) + Margin.X,
                                                                                   (float)Rand.NextDouble() * (DimsMap.Y - Margin.Y * 2) + Margin.Y,
                                                                                   (float)Rand.NextDouble() * (DimsMap.Z - Margin.Z * 2) + Margin.Z), MapSamples);

                        float3[] Angle;
                        if (Options.DontAugment)
                            Angle = Helper.ArrayOfFunction(i => new float3((float)Math.Round(Rand.NextDouble()) * 180,
                                                                           (float)Math.Round(Rand.NextDouble()) * 180,
                                                                           (float)Math.Round(Rand.NextDouble()) * 180) * Helper.ToRad, MapSamples);
                        else
                            Angle = Helper.ArrayOfFunction(i => new float3((float)Rand.NextDouble() * 360,
                                                                           (float)Rand.NextDouble() * 360,
                                                                           (float)Rand.NextDouble() * 360) * Helper.ToRad, MapSamples);

                        {
                            ulong[] Texture = new ulong[1], TextureArray = new ulong[1];
                            GPU.CreateTexture3D(Map1.GetDevice(Intent.Read), Map1.Dims, Texture, TextureArray, true);
                            Map1.FreeDevice();

                            GPU.Rotate3DExtractAt(Texture[0],
                                                  Map1.Dims,
                                                  ExtractedSource[m].GetDevice(Intent.Write),
                                                  Dim,
                                                  Helper.ToInterleaved(Angle),
                                                  Helper.ToInterleaved(Position),
                                                  (uint)MapSamples);

                            //ExtractedSource[MapID].WriteMRC("d_extractedsource.mrc", true);

                            GPU.DestroyTexture(Texture[0], TextureArray[0]);
                        }

                        {
                            ulong[] Texture = new ulong[1], TextureArray = new ulong[1];
                            GPU.CreateTexture3D(Map2.GetDevice(Intent.Read), Map2.Dims, Texture, TextureArray, true);
                            Map2.FreeDevice();

                            GPU.Rotate3DExtractAt(Texture[0],
                                                  Map2.Dims,
                                                  ExtractedTarget[m].GetDevice(Intent.Write),
                                                  Dim,
                                                  Helper.ToInterleaved(Angle),
                                                  Helper.ToInterleaved(Position),
                                                  (uint)MapSamples);

                            //ExtractedTarget.WriteMRC("d_extractedtarget.mrc", true);

                            GPU.DestroyTexture(Texture[0], TextureArray[0]);
                        }

                        {
                            for (int i = 0; i < MapSamples; i++)
                                GPU.CopyDeviceToDevice(MapCTFs[MapID].GetDevice(Intent.Read),
                                                       ExtractedCTF[m].GetDeviceSlice(i * Dim2.Z, Intent.Write),
                                                       MapCTFs[MapID].ElementsReal);
                        }

                        //Map1.FreeDevice();
                        //Map2.FreeDevice();
                    }

                    // Shuffle individual examples between batches so each batch doesn't source from only one map
                    for (int b = 0; b < MapSamples; b++)
                    {
                        int[] Order = Helper.RandomSubset(Helper.ArrayOfSequence(0, NMapsPerBatch, 1), NMapsPerBatch, Rand.Next(9999999));
                        for (int i = 0; i < Order.Length; i++)
                        {
                            GPU.CopyDeviceToDevice(ExtractedSource[i].GetDeviceSlice(b * Dim.Z, Intent.Read),
                                                   ExtractedSourceRand[Order[i]].GetDeviceSlice(b * Dim.Z, Intent.Write),
                                                   Dim.Elements());
                            GPU.CopyDeviceToDevice(ExtractedTarget[i].GetDeviceSlice(b * Dim.Z, Intent.Read),
                                                   ExtractedTargetRand[Order[i]].GetDeviceSlice(b * Dim.Z, Intent.Write),
                                                   Dim.Elements());
                            GPU.CopyDeviceToDevice(ExtractedCTF[i].GetDeviceSlice(b * Dim2.Z, Intent.Read),
                                                   ExtractedCTFRand[Order[i]].GetDeviceSlice(b * Dim2.Z, Intent.Write),
                                                   (Dim2.X / 2 + 1) * Dim2.Y * Dim2.Z);
                        }
                    }

                    Image PredictedData = null;
                    Image PredictedDataDeconv = null;
                    float[] Loss = null;

                    {
                        double CurrentLearningRate = Math.Exp(MathHelper.Lerp((float)Math.Log(Options.LearningRateStart),
                                                                              (float)Math.Log(Options.LearningRateFinish),
                                                                              iter / (float)Options.NIterations));

                        for (int m = 0; m < ShuffledMapIDs.Length; m++)
                        {
                            int MapID = m;

                            bool Twist = Rand.Next(2) == 0;

                            TrainModel.TrainDeconv((Twist ? ExtractedSourceRand : ExtractedTargetRand)[MapID],
                                                   (Twist ? ExtractedTargetRand : ExtractedSourceRand)[MapID],
                                                   ExtractedCTFRand[MapID],
                                                   (float)CurrentLearningRate,
                                                   out PredictedData,
                                                   out PredictedDataDeconv,
                                                   out Loss);

                            Losses.Enqueue(Loss[0]);
                            if (Losses.Count > 100)
                                Losses.Dequeue();

                            if (m == 0 && iter % 500 == 0)
                            {
                                ExtractedSourceRand[MapID].WriteMRC($"d_source_{iter:D6}.mrc", true);
                                PredictedData.WriteMRC($"d_predicted_{iter:D6}.mrc", true);
                                PredictedDataDeconv.WriteMRC($"d_predicteddeconv_{iter:D6}.mrc", true);
                            }
                        }
                    }

                    double TicksPerIteration = Watch.ElapsedTicks;// / (double)(iter + 1);
                    TimeSpan TimeRemaining = new TimeSpan((long)(TicksPerIteration * (Options.NIterations - 1 - iter)));

                    {
                        double CurrentLearningRate = Math.Exp(MathHelper.Lerp((float)Math.Log(Options.LearningRateStart),
                                                                              (float)Math.Log(Options.LearningRateFinish),
                                                                              iter / (float)Options.NIterations));

                        ClearCurrentConsoleLine();
                        Console.Write($"{iter + 1}/{Options.NIterations}, " +
                                      (TimeRemaining.Days > 0 ? (TimeRemaining.Days + " days ") : "") +
                                      $"{TimeRemaining.Hours}:{TimeRemaining.Minutes:D2}:{TimeRemaining.Seconds:D2} remaining, " +
                                      $"log(loss) = {Math.Log(MathHelper.Mean(Losses)).ToString("F4")}, " +
                                      $"lr = {CurrentLearningRate}, " +
                                      $"{GPU.GetFreeMemory(Options.GPUNetwork)} MB free");
                    }

                    if (float.IsNaN(Loss[0]) || float.IsInfinity(Loss[0]))
                        throw new Exception("The loss function has reached an invalid value because something went wrong during training.");

                    GPU.CheckGPUExceptions();
                    Watch.Restart();
                }

                Watch.Stop();

                NameTrainedModel = "NoiseNet3D_" + (!string.IsNullOrEmpty(Options.StartModelName) ? (Options.StartModelName + "_") : "") +
                                   DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".pt";
                TrainModel.Save(Path.Combine(WorkingDirectory, NameTrainedModel));
                TrainModel.Dispose();

                Console.WriteLine("\nDone training!\n");

                #endregion
            }

            #region Denoise

            Options.BatchSize = 2;

            Console.WriteLine("Loading trained model, " + GPU.GetFreeMemory(Options.GPUNetwork) + " MB free.");
            TrainModel = new NoiseNet3DTorch(new int3(192, 32, 192), new[] { 0, 1, 2, 3 }, Options.BatchSize);
            if (!File.Exists(Path.Combine(WorkingDirectory, NameTrainedModel)))
                throw new Exception("Old model could not be found.");
            TrainModel.Load(Path.Combine(WorkingDirectory, NameTrainedModel));
            //TrainModel = new NoiseNet3D(@"H:\denoise_refine\noisenet3d_64_20180808_010023", new int3(Dim), 1, Options.BatchSize, false, Options.GPUNetwork);
            Console.WriteLine("Loaded trained model, " + GPU.GetFreeMemory(Options.GPUNetwork) + " MB remaining.\n");

            //Directory.Delete(NameTrainedModel, true);

            Directory.CreateDirectory("denoised");

            GPU.SetDevice(Options.GPUPreprocess);

            for (int imap = 0; imap < MapsForDenoising.Count; imap++)
            {
                Console.Write($"Denoising {NamesForDenoising[imap]}... ");

                Image Map1 = MapsForDenoising[imap];
                NoiseNet3DTorch.Denoise(Map1, new NoiseNet3DTorch[] { TrainModel });

                float2 MeanStd = MeanStdForDenoising[imap];

                Map1.TransformValues(v => v * MeanStd.Y);

                if (Options.KeepDimensions)
                {
                    if (DimensionsForDenoising[imap] != Map1.Dims)
                    {
                        Image Scaled = Map1.AsScaled(DimensionsForDenoising[imap]);
                        Map1.Dispose();
                        Map1 = Scaled;
                    }
                    if (OriginalBoxForDenoising[imap] != Map1.Dims)
                    {
                        Image Padded = Map1.AsPadded(OriginalBoxForDenoising[imap]);
                        Map1.Dispose();
                        Map1 = Padded;
                    }
                }
                Map1.PixelSize = PixelSizeForDenoising[imap];

                Map1.TransformValues(v => v + MeanStd.X);

                if (Options.Lowpass > 0)
                    Map1.Bandpass(0, Map1.PixelSize * 2 / Options.Lowpass, true, 0.01f);

                if (Options.KeepDimensions && Options.MaskOutput)
                    Map1.Multiply(Mask);

                string SavePath1 = "denoised/" + NamesForDenoising[imap] + (Options.DenoiseSeparately ? "_1" : "") + ".mrc";
                Map1.WriteMRC(SavePath1, true);
                Map1.Dispose();

                Console.WriteLine("Done. Saved to " + SavePath1);

                if (Options.DenoiseSeparately)
                {
                    Console.Write($"Denoising {NamesForDenoising[imap]} (2nd observation)... ");

                    Image Map2 = MapsForDenoising2[imap];
                    NoiseNet3DTorch.Denoise(Map2, new NoiseNet3DTorch[] { TrainModel });

                    Map2.TransformValues(v => v * MeanStd.Y);

                    if (Options.KeepDimensions)
                    {
                        if (DimensionsForDenoising[imap] != Map2.Dims)
                        {
                            Image Scaled = Map2.AsScaled(DimensionsForDenoising[imap]);
                            Map2.Dispose();
                            Map2 = Scaled;
                        }
                        if (OriginalBoxForDenoising[imap] != Map2.Dims)
                        {
                            Image Padded = Map2.AsPadded(OriginalBoxForDenoising[imap]);
                            Map2.Dispose();
                            Map2 = Padded;
                        }
                    }
                    Map2.PixelSize = PixelSizeForDenoising[imap];

                    Map2.TransformValues(v => v + MeanStd.X);

                    if (Options.Lowpass > 0)
                        Map2.Bandpass(0, Map2.PixelSize * 2 / Options.Lowpass, true, 0.01f);

                    if (Options.KeepDimensions && Options.MaskOutput)
                        Map2.Multiply(Mask);

                    string SavePath2 = "denoised/" + NamesForDenoising[imap] + "_2" + ".mrc";
                    Map2.WriteMRC(SavePath2, true);
                    Map2.Dispose();

                    Console.WriteLine("Done. Saved to " + SavePath2);
                }
            }

            Console.WriteLine("\nAll done!");

            #endregion
        }

        private static void ClearCurrentConsoleLine()
        {
            int currentLineCursor = Console.CursorTop;
            Console.SetCursorPosition(0, Console.CursorTop);
            Console.Write(new string(' ', Console.WindowWidth));
            Console.SetCursorPosition(0, currentLineCursor);
        }
    }
}
