
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TorchSharp;
using TorchSharp.NN;
using TorchSharp.Tensor;
using Warp;
using Warp.NNModels;
using Warp.Tools;

namespace GANRecon
{
    class GANRecon
    {


        /*static TorchTensor relspaceProject(TorchTensor volume, TorchTensor angles, TorchTensor coordinates)
        {
            

        
        }*/

        static void Main(string[] args)
        {
            int boxLength = 96;
            int originalLength = -1;
            int2 boxsize = new(boxLength);
            int[] devices = { 0 };
            GPU.SetDevice(devices[0]);
            int batchSize = 4;
            int numEpochs = 1000;
            int discIters = 8;
            int PreProcessingDevice = 0;
            int DimRaw = 128;
            int Dim_zoom = 96;
            int Dim = 32;
            string WorkingDirectory = @"D:\GAN_recon_polcompl\";
            float3[] RandomParticleAngles = Helper.GetHealpixAngles(3).Select(s => s * Helper.ToRad).ToArray();
            {

                GPU.SetDevice(PreProcessingDevice);
                Image TrefVolume = Image.FromFile(Path.Combine(WorkingDirectory, "cryosparc_P243_J525_003_volume_map.mrc"));

                TrefVolume = TrefVolume.AsRegion(new int3((DimRaw - Dim_zoom) / 2), new int3(Dim_zoom));

                if (Dim != Dim_zoom)
                    TrefVolume = TrefVolume.AsScaled(new int3(Dim));

                TrefVolume.MaskSpherically(Dim / 2 + 2 * Dim / 8, Dim / 8, true);
                var tensorRefVolume = TensorExtensionMethods.ToTorchTensor(TrefVolume.GetHostContinuousCopy(), new long[] { 1, 1, Dim, Dim, Dim }).ToDevice(TorchSharp.DeviceType.CUDA, PreProcessingDevice);

                using (TorchTensor projMask = Float32Tensor.Ones(new long[] { 1, Dim, Dim }, DeviceType.CUDA, PreProcessingDevice))
                {

                    Image Mask = new Image(new int3(Dim, Dim, 1));
                    Mask.Fill(1);
                    //Mask.MaskSpherically(Dim / 2 + 2 * Dim / 8, Dim / 8, false);

                    GPU.CopyDeviceToDevice(Mask.GetDevice(Intent.Read), projMask.DataPtr(), Mask.ElementsReal);

                    Mask.Dispose();

                    ReconstructionWGANGenerator gen = Modules.ReconstructionWGANGenerator(tensorRefVolume, Dim);
                    int thisBatchSize = 64;
                    TorchTensor TensorS = Float32Tensor.Ones(new long[] { 1 }, DeviceType.CUDA, PreProcessingDevice, true);

                    using (TorchTensor MaskSum = projMask.Sum())
                    using (TorchTensor angles = Float32Tensor.Zeros(new long[] { thisBatchSize, 3 }, DeviceType.CUDA, PreProcessingDevice))
                    {
                        {
                            float3[] theseAngles = RandomParticleAngles.Take(thisBatchSize).ToArray();
                            float[] theseAnglesInterleaved = Helper.ToInterleaved(theseAngles);
                            GPU.CopyHostToDevice(theseAnglesInterleaved, angles.DataPtr(), thisBatchSize * 3);


                            /*calculate stdDev of proj within circular mask projMask, then penalize deviation from 1*/
                            using (TorchTensor proj = gen.Forward(angles, 0.0d))
                            using (TorchTensor maskedProj = proj * projMask)
                            using (TorchTensor maskedProjSum = maskedProj.Sum(new long[] { 2, 3 }, true))
                            using (TorchTensor maskedProjMean = maskedProjSum / MaskSum)
                            using (TorchTensor maskedProjStd = ((maskedProj - maskedProjMean).Pow(2) * maskedProj).Sum(new long[] { 2, 3 }) / MaskSum)
                            using(TorchTensor mean = maskedProjStd.Mean())
                            using (TorchTensor rec = mean.Pow(-1))
                            {
                                //s = maskedProjStd.Mean().Detach().RequiresGrad(true);
                                GPU.CopyDeviceToDevice(rec.DataPtr(), TensorS.DataPtr(), 1);
                                GPU.CheckGPUExceptions();
                                float[] buffer = new float[1];
                                GPU.CopyDeviceToHost(TensorS.DataPtr(), buffer, 1);
                                GPU.CheckGPUExceptions();
                                Console.WriteLine($"s: {buffer[0]}");
                            }

                            {
                                float[] bufferS = new float[1];
                                GPU.CopyDeviceToHost(TensorS.DataPtr(), bufferS, 1);
                                GPU.CheckGPUExceptions();
                                Console.WriteLine($"s: {bufferS[0]}");
                            }
                         }
                        int memorySize = 100;
                        int memoryPos = 0;
                        float[] lastSValues = new float[memorySize];
                        bool full = false;
                        Optimizer optim = Optimizer.Adam(new List<TorchTensor> { TensorS }, 0.01, 1e-6);
                        for (int i = 0; i < 10; i++)
                        {
                            
                            float[] losses = new float[(int)(RandomParticleAngles.Length / thisBatchSize)];
 
                            int batchIdx = 0;
                            for (int offset = 0; RandomParticleAngles.Length - offset > thisBatchSize; offset += thisBatchSize)
                            {
                                optim.ZeroGrad();
                                float3[] theseAngles = RandomParticleAngles.Skip(offset).Take(thisBatchSize).ToArray();
                                float[] theseAnglesInterleaved = Helper.ToInterleaved(theseAngles);
                                GPU.CopyHostToDevice(theseAnglesInterleaved, angles.DataPtr(), thisBatchSize * 3);
                                

                                /*calculate stdDev of proj within circular mask projMask, then penalize deviation from 1*/
                                using (TorchTensor proj = gen.Forward_Normalized(angles, TensorS))
                                using (TorchTensor maskedProj = proj * projMask)
                                using (TorchTensor maskedProjSum = maskedProj.Sum(new long[] { 2, 3 }, true))
                                using (TorchTensor maskedProjMean = maskedProjSum / MaskSum)
                                using (TorchTensor maskedProjStd = ((maskedProj - maskedProjMean).Pow(2) * maskedProj).Sum(new long[] { 2, 3 }) / MaskSum)
                                using (TorchTensor TensorLoss = (maskedProjStd - 1).Pow(2).Mean())
                                {

                                    TensorLoss.Backward();
                                    optim.Step();

                                    float[] buffer = new float[1];
                                    GPU.CopyDeviceToHost(TensorLoss.DataPtr(), buffer, 1);
                                    float loss = buffer[0];
 
                                    GPU.CopyDeviceToHost(TensorS.DataPtr(), buffer, 1);
                                    float s = buffer[0];

                                    losses[batchIdx] = loss;
                                    lastSValues[memoryPos] = s;
                                    if(memoryPos == memorySize - 1)
                                    {
                                        full = true;
                                        memoryPos = 0;
                                    }
                                    else
                                    {
                                        memoryPos++;
                                    }
                                    Console.WriteLine($"Loss: {loss}\ts: {s}");
                                    if (full && (Math.Abs(((MathHelper.Mean(lastSValues) - s) / s)) < 1e-2))
                                    {
                                        Console.WriteLine($"Final S: {s}");
                                        return;
                                    }
                                }
                                batchIdx++;
                            }
                        }
                    }
                }



            }

        /*
        var model = new ReconstructionWGAN(new int2(boxLength), devices, batchSize);

        Image imagesReal = new Image(new int3(boxLength, boxLength, batchSize));
        Image imagesCTF = new Image(new int3(boxLength, boxLength, batchSize), true, false);
        imagesCTF.Fill(1);
        Random NoiseRand = new Random(42);
        imagesReal.TransformValues(val =>
        {
            //https://stackoverflow.com/a/218600/5012099
            double u1 = 1.0 - NoiseRand.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - NoiseRand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal = 0 + 1 * randStdNormal;
            return (float)(val + randNormal);
        });
        float[] angles = Helper.ArrayOfFunction(i => 0.0f, 3 * batchSize);
        for (int i = 0; i < 100; i++)
        {

            model.TrainDiscriminatorParticle(new float[3 * batchSize], imagesReal, imagesCTF, 1e-6f, 1e-4f, out Image prediction, out float[] lossWasserstein, out float[] lossReal, out float[] lossFake, out double gradNormDis);
            model.TrainGeneratorParticle(new float[3 * batchSize], imagesCTF, imagesReal, 1e-6f, out Image predictionGen, out Image predictionNoisyGen, out float[] lossGen, out double gradNormGen);
            prediction.Dispose();
            if (i % 10 == 0)
            {
                Console.WriteLine($"{i}%\t{lossWasserstein[0]}\t{lossReal[0]}\t{lossFake[0]}\t{gradNormDis}");
            }
            prediction.Dispose();
            predictionGen.Dispose();
            predictionNoisyGen.Dispose();


        }
        */
        Console.WriteLine($"Done");
            //var NoiseNet = new NoiseNet2DTorch(boxsize, devices, batchSize);

            //Read all particles and CTF information into memory
            //{
            //    var directory = @"D:\GANRecon";
            //    var outdir = $@"{directory}\Debug_{boxLength}";

            //    if (!Directory.Exists(outdir))
            //    {
            //        Directory.CreateDirectory(outdir);
            //    }

            //    var refVolume = Image.FromFile(@"D:\GANRecon\run_1k_unfil.mrc");
            //    originalLength = refVolume.Dims.X;
            //    var refMask = Image.FromFile(@"D:\GANRecon\mask_1k.mrc");

            //    refVolume.Multiply(refMask);
            //    refVolume = refVolume.AsScaled(new int3(boxLength));
            //    refMask = refMask.AsScaled(new int3(boxLength));
            //    float[] mask2 = Helper.ArrayOfFunction(i =>
            //    {
            //        int x = i % boxLength;
            //        int y = i / boxLength;

            //        double cutoff = 45.0 / 180.0 * boxLength;
            //        double sigma = 5.0;
            //        double r = Math.Sqrt((float)(Math.Pow(x - (double)boxLength / 2.0, 2) + Math.Pow(y - (double)boxLength / 2.0, 2)));
            //        if (r < cutoff)
            //            return 1.0f;
            //        else
            //            return (float) Math.Max(0, (Math.Cos(Math.Min(1.0 , (r - cutoff) / sigma) * Math.PI) + 1.0) * 0.5);
            //        /*if (r2 > Math.Pow(45.0 / 180.0*boxLength,2))
            //            return 0.0f;
            //        else
            //            return 1.0f;*/
            //    }

            //    , boxLength * boxLength);

            //    TorchTensor tensorRefVolume = TensorExtensionMethods.ToTorchTensor(refVolume.GetHostContinuousCopy(), new long[] { 1, 1, boxLength, boxLength, boxLength }).ToDevice(TorchSharp.DeviceType.CUDA);

            //    TorchTensor tensorMaskSlice = TensorExtensionMethods.ToTorchTensor(mask2, new long[] { 1, 1, boxLength, boxLength }).ToDevice(TorchSharp.DeviceType.CUDA);
            //    Image imageMaskSlice = new Image(new int3(boxLength, boxLength, 1));
            //    GPU.CopyDeviceToDevice(tensorMaskSlice.DataPtr(), imageMaskSlice.GetDevice(Intent.Write), imageMaskSlice.ElementsReal);
            //    {


            //        imageMaskSlice.WriteMRC($@"{outdir}\imageMaskSlice.mrc", true);

            //    }

            //    var model = new ReconstructionWGAN(new int2(boxLength), 10, devices, batchSize);
            //    model.SigmaShift = 1.0d / (32 / 2);
            //    //model.getVolume();
            //    int startEpoch = 0;
            //    if (startEpoch > 0)
            //    {
            //        model.Load($@"{outdir}\model_e{startEpoch}\model");
            //    }
            //    /*
            //    float3[] angles = Helper.ArrayOfFunction(i => new float3(0, (float)(i * 10.0 / 180.0 * Math.PI), 0), 10);
            //    float[] anglesFlat = Helper.ToInterleaved(angles);
            //    TorchTensor tensorAngles = TensorExtensionMethods.ToTorchTensor<float>(anglesFlat, new long[] { 10, 3 }).ToDevice(TorchSharp.DeviceType.CUDA);
            //    TorchTensor Projected = gen.ForwardParticle(Float32Tensor.Zeros(new long[] { 10 }, TorchSharp.DeviceType.CUDA), tensorAngles, true, 0.0d);

            //    Image imProjected = new Image(new int3(boxLength, boxLength, 10));
            //    GPU.CopyDeviceToDevice(Projected.DataPtr(), imProjected.GetDevice(Intent.Write), imProjected.ElementsReal);
            //    imProjected.WriteMRC($@"{directory}\imProjected.mrc", true);

            //    Projector proj = new(refVolume, 2);
            //    proj.ProjectToRealspace(new int2(boxLength), angles).WriteMRC($@"{directory}\imWarpProjected.mrc", true);
            //    */

            //    Star particles = new Star($@"{directory}\run_data.star");
            //    var randomSubset = particles.GetColumn("rlnRandomSubset").Select((s, i) => int.Parse(s)).ToArray();
            //    var angles = particles.GetRelionAngles();
            //    var offsets = particles.GetRelionOffsets();
            //    var Ctfs = particles.GetRelionCTF();

            //    var particlePaths = particles.GetRelionParticlePaths();

            //    var uniquePaths = Helper.ArrayOfFunction(i => particlePaths[i].Item1, particlePaths.Length).Distinct().ToArray();

            //    Dictionary<String, Image> stacksByPath = new();
            //    for (int i = 0; i < uniquePaths.Length; i++)
            //    {
            //        //if(!stacksByPath.ContainsKey(particlePaths[i].Item1))
            //        //{
            //        Image im = Image.FromFile($@"{directory}\{uniquePaths[i]}");
            //        var scaled = im.AsScaled(boxsize);
            //        scaled.FreeDevice();
            //        stacksByPath[uniquePaths[i]] = scaled;
            //        im.Dispose();
            //        //}
            //    }

            //    List<float> losses = new();
            //    {
            //        int count = 0;
            //        for (int i = 0; i < randomSubset.Length; i++)
            //        {
            //            if (true || randomSubset[i] == 1)
            //            {
            //                count++;
            //            }
            //        }

            //        float3[] SubsetAngles = new float3[count];
            //        CTF[] SubsetCTFs = new CTF[count];
            //        float3[] SubsetOffsets = new float3[count];
            //        Image[] SubsetParticles = new Image[count];

            //        for (int i = 0, j = 0; i < randomSubset.Length; i++)
            //        {
            //            if (true || randomSubset[i] == 1)
            //            {
            //                SubsetAngles[j] = angles[i] * Helper.ToRad;
            //                SubsetOffsets[j] = offsets[i];
            //                SubsetCTFs[j] = Ctfs[i];
            //                SubsetParticles[j] = stacksByPath[particlePaths[i].Item1].AsSliceXY(particlePaths[i].Item2);
            //                SubsetParticles[j].MultiplySlices(imageMaskSlice);
            //                SubsetParticles[j].ShiftSlices(new float3[] { offsets[i] * ((float)originalLength) / boxLength });
            //                j++;
            //            }
            //        }

            //        Projector proj = new Projector(refVolume, 2);
            //        Image CTFCoords = CTF.GetCTFCoords(boxLength, originalLength);
            //        Random rnd = new Random();
            //        Image[] SubsetCleanParticles = Helper.ArrayOfFunction(i =>
            //        {
            //            Image im = proj.ProjectToRealspace(boxsize, new float3[] { SubsetAngles[i] });
            //            im.Normalize();
            //            im.FreeDevice();
            //            return im;
            //        }, count);
            //        Image[] SubsetCtfs = Helper.ArrayOfFunction(i =>
            //        {
            //            Image im = new(new int3(boxLength, boxLength, 1), true);
            //            GPU.CreateCTF(im.GetDevice(Intent.Write), CTFCoords.GetDevice(Intent.Read), IntPtr.Zero, (uint)CTFCoords.ElementsSliceComplex, new CTFStruct[] { SubsetCTFs[i].ToStruct() }, false, 1);
            //            im.FreeDevice();
            //            return im;
            //        }, count);
            //        TorchTensor randVolume = Float32Tensor.Random(tensorRefVolume.Shape, TorchSharp.DeviceType.CUDA);
            //        //var gen = Modules.ReconstructionWGANGenerator(randVolume, boxLength, 10);
            //        double learningRate = 1e-1;
            //        //var optimizer = Optimizer.SGD(gen.GetParameters(), 1e-2, 0.0);
            //        if(! Directory.Exists($@"{outdir}"))
            //            Directory.CreateDirectory($@"{outdir}");
            //        for (int epoch = startEpoch>0?startEpoch+1:0; epoch < numEpochs; epoch++)
            //        {
            //            float meanDiscLoss = 0.0f;
            //            float meanRealLoss = 0.0f;
            //            float meanFakeLoss = 0.0f;
            //            float meanGenLoss = 0.0f;
            //            int discSteps = 0;
            //            int genSteps = 0;
            //            /*if (epoch > 0 && epoch % 10 == 0)
            //            {
            //                learningRate = Math.Max(learningRate / 10, 1e-6);
            //                optimizer.SetLearningRateSGD(learningRate);
            //            }*/
            //            for (int numBatch = 0; numBatch < count/batchSize; numBatch++)
            //            {
            //                //optimizer.ZeroGrad();
            //                int[] thisBatch = Helper.ArrayOfFunction(i => rnd.Next(count), batchSize);

            //                /*Image BatchParticles = Image.Stack(Helper.ArrayOfFunction(i=>SubsetParticles[thisBatch[i]],batchSize));
            //                BatchParticles.ShiftSlices(Helper.ArrayOfFunction(i => SubsetOffsets[thisBatch[i]], batchSize));
            //                BatchParticles.Normalize();
            //                var BatchCTFStructs = Helper.ArrayOfFunction(i => SubsetCTFs[thisBatch[i]], batchSize);

            //                Image BatchCTFs = new Image(new int3(boxLength * 2, boxLength * 2, batchSize), true);
            //                GPU.CreateCTF(BatchCTFs.GetDevice(Intent.Write),
            //                                CTFCoords.GetDevice(Intent.Read),
            //                                IntPtr.Zero,
            //                                (uint)CTFCoords.ElementsSliceComplex,
            //                                BatchCTFStructs.Select(p => p.ToStruct()).ToArray(),
            //                                false,
            //                                (uint)BatchParticles.Dims.Z);
            //                GPU.CheckGPUExceptions();*/
            //                var BatchAngles = Helper.ArrayOfFunction(i => SubsetAngles[thisBatch[i]], batchSize);
            //                //TorchTensor tensorAngles = TensorExtensionMethods.ToTorchTensor<float>(Helper.ToInterleaved(BatchAngles), new long[] { batchSize, 3 }).ToDevice(TorchSharp.DeviceType.CUDA);
            //                //TorchTensor tensorRotMatrix = Modules.MatrixFromAngles(tensorAngles);
            //                //TorchTensor projFake = gen.ForwardParticle(Float32Tensor.Empty(new long[] { 1 }), tensorAngles, true, 0);
            //                Image source = Image.Stack(Helper.ArrayOfFunction(i => SubsetParticles[thisBatch[i]], batchSize));
            //                Image sourceCTF = Image.Stack(Helper.ArrayOfFunction(i => SubsetCtfs[thisBatch[i]], batchSize));

            //                //source.Normalize();
            //                /*TorchTensor projReal = Float32Tensor.Empty(new long[] { batchSize, 1, boxLength, boxLength }, TorchSharp.DeviceType.CUDA);
            //                GPU.CopyDeviceToDevice(source.GetDevice(Intent.Read), projReal.DataPtr(), source.ElementsReal);
            //                projReal = projReal * tensorMaskSlice.Expand(new long[] { batchSize, -1, -1, -1 });
            //                projFake = projFake * tensorMaskSlice.Expand(new long[] { batchSize, -1, -1, -1 });
            //                TorchTensor diff = projReal - projFake;
            //                TorchTensor diffSqrd = (diff).Pow(2);
            //                TorchTensor loss = diffSqrd.Mean();
            //                loss.Backward();*/
            //                if (numBatch % discIters != 0) {
            //                    model.TrainDiscriminatorParticle(Helper.ToInterleaved(BatchAngles), source, sourceCTF, (float)0.0001, (float)2, out Image prediction, out float[] wLoss, out float[] rLoss, out float[] fLoss, out double gradNormDisc);
            //                    GPU.CheckGPUExceptions();
            //                    float discLoss = wLoss[0];
            //                    meanDiscLoss += (float)discLoss;
            //                    meanRealLoss += (float)rLoss[0];
            //                    meanFakeLoss += (float)fLoss[0];
            //                    discSteps++;
            //                    //prediction.WriteMRC($@"{outdir}\prediction_{epoch}_{numBatch}.mrc", true);
            //                    prediction.Dispose();
            //                }
            //                else
            //                {
            //                    model.TrainGeneratorParticle(Helper.ToInterleaved(BatchAngles), sourceCTF, source, (float)0.0001, out Image prediction, out Image predictionNoisy, out float[] genLoss, out double gradNormDisc);
            //                    GPU.CheckGPUExceptions();
            //                    meanGenLoss += genLoss[0];
            //                    genSteps++;
            //                    prediction.Dispose();
            //                }
            //                //prediction.WriteMRC($@"{directory}\Optimization\prediction_{epoch}_{numBatch}.mrc", true);
            //                //source.WriteMRC($@"{outdir}\source_{epoch}_{numBatch}.mrc", true);
            //                //sourceCTF.WriteMRC($@"{outdir}\sourceCTF_{epoch}_{numBatch}.mrc", true);
            //                //optimizer.Step();



            //                /*if (numBatch == 0)
            //                {
            //                    Image projectionsReal = new Image(new int3(boxLength, boxLength, batchSize));
            //                    Image projectionsFake = new Image(new int3(boxLength, boxLength, batchSize));

            //                    GPU.CopyDeviceToDevice(projReal.DataPtr(), projectionsReal.GetDevice(Intent.Write), projectionsReal.ElementsReal);
            //                    GPU.CopyDeviceToDevice(projFake.DataPtr(), projectionsFake.GetDevice(Intent.Write), projectionsFake.ElementsReal);

            //                    Image stacked = Image.Stack(new Image[] { projectionsReal, projectionsFake, source });

            //                    stacked.WriteMRC($@"{directory}\Optimization\Projections_{epoch}.mrc", true);

            //                    projectionsFake.Dispose();
            //                    projectionsReal.Dispose();
            //                    stacked.Dispose();

            //                    Image vol = new Image(new int3(boxLength));
            //                    GPU.CopyDeviceToDevice(zeroVolume.DataPtr(), vol.GetDevice(Intent.Write), vol.ElementsReal);
            //                    vol.WriteMRC($@"{directory}\Optimization\Volume_{epoch}.mrc", true);

            //                }*/

            //                //BatchParticles.Dispose();
            //                //BatchCTFs.Dispose();
            //                source.Dispose();

            //                //tensorAngles.Dispose();
            //                //tensorRotMatrix.Dispose();
            //                //projFake.Dispose();
            //                //projReal.Dispose();
            //                //diff.Dispose();
            //                //diffSqrd.Dispose();
            //                //loss.Dispose();
            //            }
            //            losses.Append(meanDiscLoss / (count / batchSize));
            //            GPU.DeviceSynchronize();
            //            GPU.CheckGPUExceptions();
            //            using (StreamWriter file = new($@"{ outdir }\log.txt", append: true))
            //            {
            //                file.WriteLineAsync($"Epoch {epoch}: Disc: { meanDiscLoss / discSteps } (r: {meanRealLoss / discSteps}, f: {meanFakeLoss / discSteps}) Gen: {meanGenLoss / genSteps}");
            //            }
            //            //Console.WriteLine($"Epoch {epoch}: Disc: { meanDiscLoss / discSteps } Gen: {meanGenLoss / genSteps}");

            //            if(epoch %50==0 || epoch == numEpochs-1)
            //                model.Save($@"{outdir}\model_e{epoch}\model");
            //        }



            //    }

            //}
        }
    }
}
