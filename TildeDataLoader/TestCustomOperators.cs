using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;
using TorchSharp.Tensor;
using TorchSharp.NN;
using static TorchSharp.NN.Modules;
using static TorchSharp.NN.Functions;
using static TorchSharp.NN.Losses;
using static TorchSharp.ScalarExtensionMethods;
using TorchSharp;
using System.Diagnostics;
using System.IO;
using Warp.Headers;
using System.Globalization;

namespace TestCustomOperators
{
    class TestCustomOperators
    {
        static void Main2(string[] args)
        {
            var directory = @"Z:\DATA\fjochhe\EMPIAR\10073\data";

            Star particles = new Star($@"{directory}\shiny_correctpaths_cleanedcorruptstacks.star");
            var randomSubset = particles.GetColumn("rlnRandomSubset").Select((s, i) => int.Parse(s)).ToArray();
            var angles = particles.GetRelionAngles();
            var offsets = particles.GetRelionOffsets();
            var Ctfs = particles.GetRelionCTF();

            var particlePaths = particles.GetRelionParticlePaths();

            var uniquePaths = Helper.ArrayOfFunction(i => particlePaths[i].Item1, particlePaths.Length).Distinct().ToArray();

            Dictionary<String, Image> stacksByPath = new();
            for (int i = 0; i < uniquePaths.Length; i++)
            {
                //if(!stacksByPath.ContainsKey(particlePaths[i].Item1))
                //{
                Image im = Image.FromFile($@"R:\10073\data\{uniquePaths[i]}");
                stacksByPath[uniquePaths[i]] = im;
                //}
            }

            {
                int count = 0;
                for (int i = 0; i < randomSubset.Length; i++)
                {
                    if (randomSubset[i] == 1)
                    {
                        count++;
                    }
                }

                float3[] SubsetAngles = new float3[count];
                CTF[] SubsetCTFs = new CTF[count];
                float3[] SubsetOffsets = new float3[count];
                Image[] SubsetParticles = new Image[count];

                for (int i = 0, j = 0; i < randomSubset.Length; i++)
                {
                    if (randomSubset[i] == 1)
                    {
                        SubsetAngles[j] = angles[i];
                        SubsetOffsets[j] = offsets[i];
                        SubsetCTFs[j] = Ctfs[i];
                        SubsetParticles[j] = stacksByPath[particlePaths[i].Item1].AsSliceXY(particlePaths[i].Item2);
                        j++;
                    }
                }

                Projector proj = new Projector(new int3(380), 1);
                Image CTFCoords = CTF.GetCTFCoords(SubsetParticles[0].Dims.X, SubsetParticles[0].Dims.X);
                for (int processIdx = 0; processIdx < SubsetParticles.Length; processIdx += 2048)
                {
                    //float[][] slicesData = particles[i].GetHost(Intent.Read).Skip(processIdx).Take(1024).ToArray();
                    //Image part = new Image(slicesData, new int3(particles[i].Dims.X, particles[i].Dims.Y, slicesData.Length));
                    Image BatchParticles = Image.Stack(SubsetParticles.Skip(processIdx).Take(1024).ToArray());
                    BatchParticles.ShiftSlices(SubsetOffsets.Skip(processIdx).Take(1024).ToArray());
                    Image BatchParticlesFT = BatchParticles.AsFFT();
                    var BatchCTFStructs = SubsetCTFs.Skip(processIdx).Take(1024).ToArray();

                    Image BatchCTFs = new Image(BatchParticles.Dims, true);
                    GPU.CreateCTF(BatchCTFs.GetDevice(Intent.Write),
                                    CTFCoords.GetDevice(Intent.Read),
                                    IntPtr.Zero,
                                    (uint)CTFCoords.ElementsSliceComplex,
                                    BatchCTFStructs.Select(p => p.ToStruct()).ToArray(),
                                    false,
                                    (uint)BatchParticles.Dims.Z);
                    GPU.CheckGPUExceptions();
                    BatchParticlesFT.ShiftSlices(Helper.ArrayOfFunction(j => new float3(BatchParticles.Dims.X / 2, BatchParticles.Dims.Y / 2, 0), BatchParticles.Dims.Z));
                    BatchParticlesFT.Multiply(BatchCTFs);
                    BatchCTFs.Multiply(BatchCTFs);
                    GPU.CheckGPUExceptions();

                    proj.BackProject(BatchParticlesFT, BatchCTFs, SubsetAngles.Skip(processIdx).Take(BatchParticles.Dims.Z).Select(a => a * Helper.ToRad).ToArray(), new float3(1, 1, 0));
                    GPU.CheckGPUExceptions();
                    BatchParticles.Dispose();
                    BatchParticlesFT.Dispose();
                    BatchCTFs.Dispose();

                }
                GPU.CheckGPUExceptions();
                Image Rec = proj.Reconstruct(false, "C1");
                Rec.WriteMRC($@"{directory}\run_half1_class001_unfil_WARP.mrc", true);
            }

            {
                int count = 0;
                for (int i = 0; i < randomSubset.Length; i++)
                {
                    if (randomSubset[i] == 2)
                    {
                        count++;
                    }
                }

                float3[] SubsetAngles = new float3[count];
                CTF[] SubsetCTFs = new CTF[count];
                float3[] SubsetOffsets = new float3[count];
                Image[] SubsetParticles = new Image[count];

                for (int i = 0, j = 0; i < randomSubset.Length; i++)
                {
                    if (randomSubset[i] == 2)
                    {
                        SubsetAngles[j] = angles[i];
                        SubsetOffsets[j] = offsets[i];
                        SubsetCTFs[j] = Ctfs[i];
                        SubsetParticles[j] = stacksByPath[particlePaths[i].Item1].AsSliceXY(particlePaths[i].Item2);
                        j++;
                    }
                }

                Projector proj = new Projector(new int3(380), 1);
                Image CTFCoords = CTF.GetCTFCoords(SubsetParticles[0].Dims.X, SubsetParticles[0].Dims.X);
                for (int processIdx = 0; processIdx < SubsetParticles.Length; processIdx += 2048)
                {
                    //float[][] slicesData = particles[i].GetHost(Intent.Read).Skip(processIdx).Take(1024).ToArray();
                    //Image part = new Image(slicesData, new int3(particles[i].Dims.X, particles[i].Dims.Y, slicesData.Length));
                    Image BatchParticles = Image.Stack(SubsetParticles.Skip(processIdx).Take(1024).ToArray());
                    BatchParticles.ShiftSlices(SubsetOffsets.Skip(processIdx).Take(1024).ToArray());
                    Image BatchParticlesFT = BatchParticles.AsFFT();
                    var BatchCTFStructs = SubsetCTFs.Skip(processIdx).Take(1024).ToArray();

                    Image BatchCTFs = new Image(BatchParticles.Dims, true);
                    GPU.CreateCTF(BatchCTFs.GetDevice(Intent.Write),
                                    CTFCoords.GetDevice(Intent.Read),
                                    IntPtr.Zero,
                                    (uint)CTFCoords.ElementsSliceComplex,
                                    BatchCTFStructs.Select(p => p.ToStruct()).ToArray(),
                                    false,
                                    (uint)BatchParticles.Dims.Z);
                    GPU.CheckGPUExceptions();
                    BatchParticlesFT.ShiftSlices(Helper.ArrayOfFunction(j => new float3(BatchParticles.Dims.X / 2, BatchParticles.Dims.Y / 2, 0), BatchParticles.Dims.Z));
                    BatchParticlesFT.Multiply(BatchCTFs);
                    BatchCTFs.Multiply(BatchCTFs);
                    GPU.CheckGPUExceptions();

                    proj.BackProject(BatchParticlesFT, BatchCTFs, SubsetAngles.Skip(processIdx).Take(BatchParticles.Dims.Z).Select(a => a * Helper.ToRad).ToArray(), new float3(1, 1, 0));
                    GPU.CheckGPUExceptions();
                    BatchParticles.Dispose();
                    BatchParticlesFT.Dispose();
                    BatchCTFs.Dispose();

                }
                GPU.CheckGPUExceptions();
                Image Rec = proj.Reconstruct(false, "C1");
                Rec.WriteMRC($@"{directory}\run_half2_class001_unfil_WARP.mrc", true);
            }
        }

        static void testFFTCropSpeed()
        {
            //Test FFT Crop speed
            if (true)
            {
                TorchTensor fullVolume = Float32Tensor.Random(new long[] { 20, 100, 100, 100 }, DeviceType.CUDA);
                TorchTensor fullFFT = fullVolume.rfftn(new long[] { 1, 2, 3 });
                for (int j = 0; j < 10; j++)
                {
                    Stopwatch stopWatch = new Stopwatch();
                    stopWatch.Start();
                    for (int i = 0; i < 1000; i++)
                    {
                        TorchTensor detached = fullFFT.Detach();
                        TorchTensor thisFullFFT = detached.RequiresGrad(true);
                        TorchTensor newCroppedFFT = FFTCrop(thisFullFFT, 3, 50, 50, 50);
                        TorchTensor abs = newCroppedFFT.Abs();
                        TorchTensor mean = abs.Mean();
                        //mean.Backward();
                        newCroppedFFT.Dispose();
                        mean.Dispose();
                        abs.Dispose();
                        thisFullFFT.Dispose();
                        detached.Dispose();

                    }
                    stopWatch.Stop();
                    // Get the elapsed time as a TimeSpan value.
                    TimeSpan ts = stopWatch.Elapsed;

                    // Format and display the TimeSpan value.
                    string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                        ts.Hours, ts.Minutes, ts.Seconds,
                        ts.Milliseconds / 10);
                    Console.WriteLine("RunTime new method " + elapsedTime);
                }
                /* I deleted the method based on my own kernel. It was about 4 times slower
                for (int j = 0; j < 10; j++)
                {
                    Stopwatch stopWatch = new Stopwatch();
                    stopWatch.Start();

                    for (int i = 0; i < 1000; i++)
                    {

                        TorchTensor detached = fullFFT.Detach();
                        TorchTensor thisFullFFT = detached.RequiresGrad(true);
                        TorchTensor newCroppedFFT = FFTCrop(thisFullFFT, 50, 50, 50);
                        TorchTensor abs = newCroppedFFT.Abs();
                        TorchTensor mean = abs.Mean();
                        //mean.Backward();
                        mean.Dispose();
                        detached.Dispose();
                        abs.Dispose();
                        newCroppedFFT.Dispose();
                        mean.Dispose();
                        thisFullFFT.Dispose();
                    }
                    stopWatch.Stop();
                    // Get the elapsed time as a TimeSpan value.
                    TimeSpan ts = stopWatch.Elapsed;

                    // Format and display the TimeSpan value.
                    string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                        ts.Hours, ts.Minutes, ts.Seconds,
                        ts.Milliseconds / 10);
                    Console.WriteLine("RunTime old method " + elapsedTime);
                }
                */
                {
                    //TorchTensor croppedFFT = FFTCrop(fullFFT, 50, 50, 50);
                    //TorchTensor newPaddedFFT = newFFTCrop(fullFFT, 3, 150, 150, 150);
                    //TorchTensor newCroppedFFT = newFFTCrop(newPaddedFFT, 3, 50, 50, 50);

                    //bool test = croppedFFT.AllClose(newCroppedFFT);
                }
            }
        }
        static void testFFTCrop()
        {
            //Test FFT Crop vs warp crop
            {
                Image sourceVolumeOrg = Image.FromFile(@"D:\EMD\9233\emd_9233_2.0.mrc");
                int cropDim = 100;

                //Warp based Fourier Cropping for reference
                Image WarpCropped = sourceVolumeOrg.AsScaled(new int3(100));

                // Create a cropped volume using Torch Operator
                TorchTensor TensorFullVolume = Float32Tensor.Zeros(new long[] { 1, sourceVolumeOrg.Dims.Z, sourceVolumeOrg.Dims.Y, sourceVolumeOrg.Dims.X }, DeviceType.CUDA, 0);
                GPU.CopyDeviceToDevice(sourceVolumeOrg.GetDevice(Intent.Read), TensorFullVolume.DataPtr(), sourceVolumeOrg.ElementsReal);
                GPU.CheckGPUExceptions();
                TorchTensor TensorFFTFullVolume = TensorFullVolume.rfftn(new long[] { 1, 2, 3 });
                TorchTensor TensorFFTCroppedVolume = FFTCrop(TensorFFTFullVolume, 3, cropDim, cropDim, cropDim);
                TorchTensor TensorCroppedVolume = TensorFFTCroppedVolume.irfftn(new long[] { 1, 2, 3 });

                TorchTensor TensorWarpCropped = Float32Tensor.Zeros(new long[] { 1, cropDim, cropDim, cropDim }, DeviceType.CUDA, 0);
                GPU.CopyDeviceToDevice(WarpCropped.GetDevice(Intent.Read), TensorWarpCropped.DataPtr(), WarpCropped.ElementsReal);

                bool isClose = TensorCroppedVolume.AllClose(TensorWarpCropped, 1E-5, 1E-5);
                Debug.Assert(isClose, "Warp and torch cropping are not the same");
                {
                    Image croppedImage = new(new int3(cropDim));
                    GPU.CopyDeviceToDevice(TensorWarpCropped.DataPtr(), croppedImage.GetDevice(Intent.Write), croppedImage.ElementsReal);
                    croppedImage.WriteMRC("WarpCropped.mrc", true);
                }
                {
                    Image croppedImage = new(new int3(cropDim));
                    GPU.CopyDeviceToDevice(TensorCroppedVolume.DataPtr(), croppedImage.GetDevice(Intent.Write), croppedImage.ElementsReal);
                    croppedImage.WriteMRC("OwnCropped.mrc", true);
                }
            }
        }

        static int makeEven(int num)
        {
            return (num / 2) * 2;
        }

        static void testVolumeScaling()
        {
            //Test FFT Crop vs warp crop
            {
                Image sourceVolumeOrg = Image.FromFile(@"D:\EMD\9233\emd_9233_2.0.mrc");
                sourceVolumeOrg.AsFFT(true).AsAmplitudes().WriteMRC("SourceVolumeFFT.mrc", true);
                TorchTensor TensorFullVolume = Float32Tensor.Zeros(new long[] { 1, sourceVolumeOrg.Dims.Z, sourceVolumeOrg.Dims.Y, sourceVolumeOrg.Dims.X }, DeviceType.CUDA, 0);
                GPU.CopyDeviceToDevice(sourceVolumeOrg.GetDevice(Intent.Read), TensorFullVolume.DataPtr(), sourceVolumeOrg.ElementsReal);
                GPU.CheckGPUExceptions();
                int cropDim = makeEven(sourceVolumeOrg.Dims.X/2);
                int padDim = makeEven(sourceVolumeOrg.Dims.X*2);


                //Cropping
                {
                    //Warp based Fourier Cropping for reference
                    Image WarpCroppedVolume = sourceVolumeOrg.AsScaled(new int3(cropDim));
                    TorchTensor TensorWarpCroppedVolume = Float32Tensor.Zeros(new long[] { 1, cropDim, cropDim, cropDim }, DeviceType.CUDA, 0);
                    GPU.CopyDeviceToDevice(WarpCroppedVolume.GetDevice(Intent.Read), TensorWarpCroppedVolume.DataPtr(), WarpCroppedVolume.ElementsReal);

                    // Create a cropped volume using Torch Operator
                    TorchTensor TensorCroppedVolume = ScaleVolume(TensorFullVolume, 3, cropDim, cropDim, cropDim);

                    //Write Out resulting volumes
                    WarpCroppedVolume.WriteMRC("WarpCroppedVolume.mrc", true);
                    Image OwnCroppedVolume = new(new int3(cropDim));
                    GPU.CopyDeviceToDevice(TensorCroppedVolume.DataPtr(), OwnCroppedVolume.GetDevice(Intent.Write), OwnCroppedVolume.ElementsReal);
                    OwnCroppedVolume.WriteMRC("OwnCroppedVolume.mrc", true);

                    //Check for difference
                    bool isClose = TensorCroppedVolume.AllClose(TensorWarpCroppedVolume, 1E-5, 1E-5);
                    Debug.Assert(isClose, "Warp and torch cropping are not the same");
                }

                //Cropping Projections
                {
                    Projector proj = new Projector(sourceVolumeOrg, 2);
                    Image projected = proj.ProjectToRealspace(new int2(sourceVolumeOrg.Dims.X), Helper.ArrayOfFunction(i => new float3((float)(i * 20.0 / 180.0 * Math.PI)), 18));

                    Image WarpCroppedProjections = projected.AsScaled(new int2(cropDim));
                    TorchTensor TensorProjected = Float32Tensor.Empty(new long[] {1, 18, sourceVolumeOrg.Dims.Y, sourceVolumeOrg.Dims.X }, DeviceType.CUDA, 0);
                    GPU.CopyDeviceToDevice(projected.GetDevice(Intent.Read), TensorProjected.DataPtr(), projected.ElementsReal);

                    TorchTensor TensorCroppedProjected = ScaleVolume(TensorProjected, 2, cropDim, cropDim, -1);

                    //Write Out resulting volumes
                    WarpCroppedProjections.WriteMRC("WarpCroppedProjections.mrc", true);
                    Image OwnCroppedProjections = new(WarpCroppedProjections.Dims);
                    GPU.CopyDeviceToDevice(TensorCroppedProjected.DataPtr(), OwnCroppedProjections.GetDevice(Intent.Write), OwnCroppedProjections.ElementsReal);
                    OwnCroppedProjections.WriteMRC("OwnCroppedProjections.mrc", true);


                }



                //Padding
                {
                    //Warp based Fourier Cropping for reference
                    Image WarpPaddedVolume = sourceVolumeOrg.AsScaled(new int3(padDim));
                    TorchTensor TensorWarpPaddedVolume = Float32Tensor.Zeros(new long[] { 1, padDim, padDim, padDim }, DeviceType.CUDA, 0);
                    GPU.CopyDeviceToDevice(WarpPaddedVolume.GetDevice(Intent.Read), TensorWarpPaddedVolume.DataPtr(), WarpPaddedVolume.ElementsReal);

                    // Create a cropped volume using Torch Operator
                    TorchTensor TensorPaddedVolume = ScaleVolume(TensorFullVolume, 3, padDim, padDim, padDim);

                    //Write Out resulting volumes
                    WarpPaddedVolume.WriteMRC("WarpPaddedVolume.mrc", true);
                    Image WarpPaddedVolumeFFT = WarpPaddedVolume.AsFFT(true);
                    WarpPaddedVolumeFFT = WarpPaddedVolumeFFT.AsAmplitudes();
                    WarpPaddedVolumeFFT.WriteMRC("WarpPaddedVolumeFFT.mrc", true);
                    Image OwnPaddedVolume = new(new int3(padDim));
                    GPU.CopyDeviceToDevice(TensorPaddedVolume.DataPtr(), OwnPaddedVolume.GetDevice(Intent.Write), OwnPaddedVolume.ElementsReal);
                    OwnPaddedVolume.WriteMRC("OwnPaddedVolume.mrc", true);
                    Image OwnPaddedVolumeFFT = OwnPaddedVolume.AsFFT(true);
                    OwnPaddedVolumeFFT = OwnPaddedVolumeFFT.AsAmplitudes();
                    OwnPaddedVolumeFFT.WriteMRC("OwnPaddedVolumeFFT.mrc", true);
                    OwnPaddedVolume.Subtract(WarpPaddedVolume);
                    OwnPaddedVolume.Abs();
                    OwnPaddedVolume.WriteMRC("DIFFReal.mrc", true);
                    OwnPaddedVolumeFFT.Subtract(WarpPaddedVolumeFFT);
                    OwnPaddedVolumeFFT.Abs();
                    OwnPaddedVolumeFFT.WriteMRC("FFTReal.mrc", true);
                    //Check for difference
                    bool isClose = TensorPaddedVolume.AllClose(TensorWarpPaddedVolume, 1E-5, 1E-5);
                    Debug.Assert(isClose, "Warp and torch cropping are not the same");


                }

            }
        }


        static void Main3(string[] args)
        {
            GPU.SetDevice(0);
            Image sourceVolumeOrg = Image.FromFile(@"D:\EMD\9233\emd_9233_2.0.mrc");

            //Warp based Fourier Cropping for reference
            Image sourceVolume = sourceVolumeOrg.AsScaled(new int3(100));
            sourceVolume.WriteMRC("WarpCropped.mrc", true);

            // Create a cropped volume using Torch Operator
            TorchTensor TensorFullVolume = Float32Tensor.Random(new long[] { 1, sourceVolumeOrg.Dims.Z, sourceVolumeOrg.Dims.Y, sourceVolumeOrg.Dims.X }, DeviceType.CUDA, 0);
            GPU.CopyDeviceToDevice(sourceVolumeOrg.GetDevice(Intent.Read), TensorFullVolume.DataPtr(), sourceVolumeOrg.ElementsReal);
            TorchTensor TensorFFTFullVolume = TensorFullVolume.rfftn(new long[] { 1, 2, 3 });
            TorchTensor TensorFFTCroppedVolume = FFTCrop(TensorFFTFullVolume, 3, 100, 100, 100);
            TorchTensor TensorCroppedVolume = TensorFFTCroppedVolume.irfftn(new long[] { 1, 2, 3 });
            Image croppedImage = new(new int3(100, 100, 100));
            GPU.CopyDeviceToDevice(TensorCroppedVolume.DataPtr(), croppedImage.GetDevice(Intent.Write), croppedImage.ElementsReal);
            croppedImage.WriteMRC("OwnCropped.mrc", true);


            //Now make the test if the gradient is calculated correctly. Create an empty image and calculate updates for it after doing the FFT crop
            TorchTensor RandomImg = Float32Tensor.Zeros(new long[] { 1, sourceVolumeOrg.Dims.Z, sourceVolumeOrg.Dims.Y, sourceVolumeOrg.Dims.X }, DeviceType.CUDA, 0, true);
            var optimizer = Optimizer.SGD(new List<TorchTensor> { RandomImg }, 0.01);

            for (int i = 0; i < 1000; i++)
            {
                optimizer.ZeroGrad();

                TorchTensor fft = RandomImg.rfftn(new long[] { 1, 2, 3 });

                TorchTensor croppedfft = FFTCrop(fft, 3, 100, 100, 100);

                TorchTensor cropped = croppedfft.irfftn(new long[] { 1, 2, 3 });
                TorchTensor diff = cropped - TensorCroppedVolume;
                TorchTensor diffSqrd = diff.Pow(2);

                TorchTensor error = diffSqrd.Mean();
                error.Backward();
                optimizer.Step();

                fft.Dispose();
                croppedfft.Dispose();
                cropped.Dispose();
                diff.Dispose();
                diffSqrd.Dispose();
                error.Dispose();
                if (i % 100 == 0)
                {
                    Image backwardsImage = new(sourceVolumeOrg.Dims);
                    GPU.CopyDeviceToDevice(RandomImg.DataPtr(), backwardsImage.GetDevice(Intent.Write), backwardsImage.ElementsReal);
                    backwardsImage.WriteMRC($"OwnCroppedBackwards_{i}.mrc", true);
                }
            }
            {
                Image backwardsImage = new(sourceVolumeOrg.Dims);
                GPU.CopyDeviceToDevice(RandomImg.DataPtr(), backwardsImage.GetDevice(Intent.Write), backwardsImage.ElementsReal);
                backwardsImage.WriteMRC("OwnCroppedBackwards.mrc", true);
                {
                    {
                        Image backwardsImFFT = backwardsImage.AsFFT();
                        Image backwardsAbsFFT = new(new int3(134), true);
                        GPU.Amplitudes(backwardsImFFT.GetDevice(Intent.Read), backwardsAbsFFT.GetDevice(Intent.Write), backwardsImFFT.ElementsComplex);
                        backwardsAbsFFT.WriteMRC("OwnCroppedBackwards_FFT.mrc", true);
                    }
                }
            }
        }

        static void TestAtomRastering()
        {
            GPU.SetDevice(0);
            int dim = 20;
            int numAtoms = 125;
            int numOrientations = 2;
            //Create random atom positions for x many atoms



            TorchTensor atomIntensities = Float32Tensor.Random(new long[] { 1, numAtoms }, DeviceType.CUDA);
            TorchTensor atomPositions = Float32Tensor.Zeros(new long[] { numOrientations, numAtoms, 3 }, DeviceType.CUDA);
            var rand = new Random(42);
            {
                float[] h_atomPositions = new float[125 * 3 * numOrientations];
                int i = 0;
                for (int zz = 8; zz < 13; zz++)
                {
                    for (int yy = 8; yy < 13; yy++)
                    {
                        for (int xx = 8; xx < 13; xx++)
                        {
                            float x = (float)((float)xx + rand.NextDouble() - 0.5) / (dim - 1) * 2 - 1;
                            float y = (float)((float)yy + rand.NextDouble() - 0.5) / (dim - 1) * 2 - 1;
                            float z = (float)((float)zz + rand.NextDouble() - 0.5) / (dim - 1) * 2 - 1;
                            for (int j = 0; j < numOrientations; j++)
                            {


                                h_atomPositions[125 * 3 * j + i] = x;

                                h_atomPositions[125 * 3 * j + i + 1] = y;

                                h_atomPositions[125 * 3 * j + i + 2] = z;
                            }
                            i += 3;
                        }
                    }
                }
                GPU.CopyHostToDevice(h_atomPositions, atomPositions.DataPtr(), numOrientations * numAtoms * 3);


                GPU.CheckGPUExceptions();
            }

            //Create rotations of let's say 1 different angles
            TorchTensor projectionOrientations = Float32Tensor.Zeros(new long[] { numOrientations, 3, 3 }, DeviceType.CUDA);
            {
                float[] h_data = Helper.ArrayOfFunction(i =>
                {
                    Matrix3 rotation = Matrix3.Euler((float)(30 * Math.PI / 180), 0, 0);
                    return rotation.ToArray()[i % 9];
                }, 9 * numOrientations);
                GPU.CopyHostToDevice(h_data, projectionOrientations.DataPtr(), 9 * numOrientations);
                GPU.CheckGPUExceptions();
            }

            //Project ground truth, i.e. the true atom positions with the different angles

            var proj = AtomProjector(atomIntensities, dim, dim, dim);
            TorchTensor GT_Shift = Float32Tensor.Zeros(new long[] { numOrientations, 3 }, DeviceType.CUDA);


            //Create random angles, project real positions and try to get actual angles
            if (false)
            {
                TorchTensor GT_Projections = proj.ProjectToPlane(atomPositions, projectionOrientations, GT_Shift);
                GPU.CheckGPUExceptions();
                {
                    Image GT_ProjectionsImage = new(new int3(dim, dim, numOrientations));
                    GPU.CopyDeviceToDevice(GT_Projections.DataPtr(), GT_ProjectionsImage.GetDevice(Intent.Write), GT_ProjectionsImage.ElementsReal);
                    GPU.CheckGPUExceptions();
                    GT_ProjectionsImage.WriteMRC($"GT_ProjectionsImage.mrc", true);
                }
                TorchTensor Random_ProjectionOrientations = Float32Tensor.Zeros(new long[] { numOrientations, 3, 3 }, DeviceType.CUDA, 0, true);
                {
                    float[] h_data = Helper.ArrayOfFunction(i =>
                    {

                        Matrix3 rotation = Matrix3.Euler((float)(0 * Math.PI / 180), 0, 0);
                        return rotation.ToArray()[i % 9];
                    }, 9 * numOrientations);
                    GPU.CopyHostToDevice(h_data, Random_ProjectionOrientations.DataPtr(), 9 * numOrientations);
                    GPU.CheckGPUExceptions();
                }
                Optimizer optim = Optimizer.Adam(new List<TorchTensor> { Random_ProjectionOrientations }, 0.001, 0);

                int numIt = 1000;
                bool printMAtrix = false;
                bool WriteImage = false;
                if (printMAtrix)
                {
                    float[] h_neworientations = new float[9 * numOrientations];
                    GPU.CopyDeviceToHost(projectionOrientations.DataPtr(), h_neworientations, 9 * numOrientations);
                    GPU.CheckGPUExceptions();
                    for (int j = 0; j < numOrientations; j++)
                    {
                        Console.WriteLine($"GTMatrix {j} :\n{h_neworientations[9 * j + 0]}\t{h_neworientations[9 * j + 1]}\t{h_neworientations[9 * j + 2]}\n{h_neworientations[9 * j + 3]}\t{h_neworientations[9 * j + 4]}\t{h_neworientations[9 * j + 5]}\n{h_neworientations[9 * j + 6]}\t{h_neworientations[9 * j + 7]}\t{h_neworientations[9 * j + 8]}");
                    }
                }
                for (int i = 0; i < numIt; i++)
                {
                    optim.ZeroGrad();
                    TorchTensor theseAtomPositions = atomPositions.Detach().RequiresGrad(true);
                    TorchTensor Pred_Projections = proj.ProjectToPlane(theseAtomPositions, Random_ProjectionOrientations, GT_Shift.Detach());

                    if ((i == 0 || i == numIt - 1) || WriteImage)
                    {
                        Image Pred_ProjectionsImage = new(new int3(dim, dim, numOrientations));
                        GPU.CopyDeviceToDevice(Pred_Projections.DataPtr(), Pred_ProjectionsImage.GetDevice(Intent.Write), Pred_ProjectionsImage.ElementsReal);
                        Pred_ProjectionsImage.WriteMRC($"Pred_ProjectionsImage_{i}_{numOrientations}.mrc", true);


                    }
                    TorchTensor error = Pred_Projections - GT_Projections.Detach();
                    TorchTensor errorSqrd = error.Pow(2);
                    TorchTensor loss = numOrientations * errorSqrd.Mean();
                    Console.WriteLine($"Loss:{loss.Detach().ToDevice(DeviceType.CPU).ToDouble()}");
                    loss.Backward();

                    if (printMAtrix)
                    {
                        TorchTensor predProj = Pred_Projections.Detach().RequiresGrad(true);
                        TorchTensor error_detach = predProj - GT_Projections.Detach();
                        TorchTensor errorSqrd_detach = error_detach.Pow(2);
                        TorchTensor loss_detach = numOrientations * errorSqrd_detach.Mean();
                        loss_detach.Backward();
                        Image Pred_ProjectionsImage = new(new int3(dim, dim, numOrientations));
                        GPU.CopyDeviceToDevice(predProj.Grad().DataPtr(), Pred_ProjectionsImage.GetDevice(Intent.Write), Pred_ProjectionsImage.ElementsReal);
                        Pred_ProjectionsImage.WriteMRC($"Pred_ProjectionsImage_Grad_{i}_{numOrientations}.mrc", true);
                    }

                    if (printMAtrix)
                    {
                        float[] h_neworientations = new float[9 * numOrientations];
                        GPU.CopyDeviceToHost(Random_ProjectionOrientations.DataPtr(), h_neworientations, 9 * numOrientations);
                        for (int j = 0; j < numOrientations; j++)
                        {
                            Console.WriteLine($"OldMatrix {i}.{j} :\n{h_neworientations[9 * j + 0]}\t{h_neworientations[9 * j + 1]}\t{h_neworientations[9 * j + 2]}\n{h_neworientations[9 * j + 3]}\t{h_neworientations[9 * j + 4]}\t{h_neworientations[9 * j + 5]}\n{h_neworientations[9 * j + 6]}\t{h_neworientations[9 * j + 7]}\t{h_neworientations[9 * j + 8]}");
                        }
                    }
                    if (printMAtrix)
                    {
                        float[] h_neworientations = new float[9 * numOrientations];
                        GPU.CopyDeviceToHost(Random_ProjectionOrientations.Grad().DataPtr(), h_neworientations, 9 * numOrientations);
                        for (int j = 0; j < numOrientations; j++)
                        {
                            Console.WriteLine($"Gradient {i}.{j} :\n{h_neworientations[9 * j + 0]}\t{h_neworientations[9 * j + 1]}\t{h_neworientations[9 * j + 2]}\n{h_neworientations[9 * j + 3]}\t{h_neworientations[9 * j + 4]}\t{h_neworientations[9 * j + 5]}\n{h_neworientations[9 * j + 6]}\t{h_neworientations[9 * j + 7]}\t{h_neworientations[9 * j + 8]}");
                        }

                    }
                    if (printMAtrix)
                    {
                        float[] h_gradPositions = new float[125 * numOrientations * 3];
                        GPU.CopyDeviceToHost(theseAtomPositions.Grad().DataPtr(), h_gradPositions, 125 * numOrientations * 3);
                        for (int j = 0; j < numOrientations; j++)
                        {
                            for (int k = 0; k < 125; k++)
                            {
                                Console.WriteLine($"Gradient {i}.{j}.{k} :{h_gradPositions[125 * j + 3 * k + 0]}\t{h_gradPositions[125 * j + 3 * k + 1]}\t{h_gradPositions[125 * j + 3 * k + 2]}");
                            }
                        }

                    }
                    optim.Step();
                    if (printMAtrix)
                    {
                        float[] h_neworientations = new float[9 * numOrientations];
                        GPU.CopyDeviceToHost(Random_ProjectionOrientations.DataPtr(), h_neworientations, 9 * numOrientations);
                        for (int j = 0; j < numOrientations; j++)
                        {
                            Console.WriteLine($"NewMatrix {i}.{j} :\n{h_neworientations[9 * j + 0]}\t{h_neworientations[9 * j + 1]}\t{h_neworientations[9 * j + 2]}\n{h_neworientations[9 * j + 3]}\t{h_neworientations[9 * j + 4]}\t{h_neworientations[9 * j + 5]}\n{h_neworientations[9 * j + 6]}\t{h_neworientations[9 * j + 7]}\t{h_neworientations[9 * j + 8]}");
                        }
                    }
                    Pred_Projections.Dispose();
                    error.Dispose();
                    errorSqrd.Dispose();
                    loss.Dispose();

                }
            }

            //Introduce a random shift globally to all atoms, then try to find that shift
            if (false)
            {
                TorchTensor GT_Projections = proj.ProjectToPlane(atomPositions, projectionOrientations, GT_Shift);
                GPU.CheckGPUExceptions();
                {
                    Image GT_ProjectionsImage = new(new int3(dim, dim, numOrientations));
                    GPU.CopyDeviceToDevice(GT_Projections.DataPtr(), GT_ProjectionsImage.GetDevice(Intent.Write), GT_ProjectionsImage.ElementsReal);
                    GPU.CheckGPUExceptions();
                    GT_ProjectionsImage.WriteMRC($"GT_ProjectionsImage.mrc", true);
                }

                float[] h_shift = new float[numOrientations * 3];
                for (int i = 0; i < numOrientations * 3; i++)
                {
                    h_shift[i] = (float)(0.02);
                }
                TorchTensor randomShift = Float32Tensor.Zeros(new long[] { numOrientations, 3 }, DeviceType.CUDA);
                GPU.CopyHostToDevice(h_shift, randomShift.DataPtr(), numOrientations * 3);
                randomShift = randomShift.Detach().RequiresGrad(true);

                Optimizer optim = Optimizer.SGD(new List<TorchTensor> { randomShift }, 0.1);

                int numIt = 10;

                for (int i = 0; i < numIt; i++)
                {
                    optim.ZeroGrad();
                    TorchTensor thisAtomPositions = atomPositions.Detach();
                    TorchTensor thisProjectionOrientations = projectionOrientations.Detach();

                    TorchTensor projected = proj.ProjectToPlane(thisAtomPositions, thisProjectionOrientations, randomShift);

                    TorchTensor error = projected - GT_Projections.Detach();
                    TorchTensor error_sqr = error.Pow(2);
                    TorchTensor loss = error_sqr.Mean();
                    Console.WriteLine($"Loss:{loss.Detach().ToDevice(DeviceType.CPU).ToDouble()}");
                    loss.Backward();
                    if (false)
                    {
                        {
                            Image Pred_ProjectionsImage = new(new int3(dim, dim, numOrientations));
                            GPU.CopyDeviceToDevice(projected.DataPtr(), Pred_ProjectionsImage.GetDevice(Intent.Write), Pred_ProjectionsImage.ElementsReal);
                            Pred_ProjectionsImage.WriteMRC($"Shifted_ProjectionsImage_{i}_{numOrientations}.mrc", true);


                        }
                        Console.WriteLine("Grad Shift:");
                        float[] h_gradShift = new float[numOrientations * 3];
                        GPU.CopyDeviceToHost(randomShift.Grad().DataPtr(), h_gradShift, numOrientations * 3);
                        for (int k = 0; k < numOrientations * 3; k++)
                        {
                            Console.Write($"{h_gradShift[k]}\t");
                            if (k % 3 == 2)
                            {
                                Console.WriteLine("");
                            }
                        }
                        Console.WriteLine("");
                    }

                    optim.Step();

                    if (false)
                    {
                        Console.WriteLine("New Shift:");
                        float[] h_randShift = new float[numOrientations * 3];
                        GPU.CopyDeviceToHost(randomShift.DataPtr(), h_randShift, numOrientations * 3);
                        for (int k = 0; k < numOrientations * 3; k++)
                        {
                            Console.Write($"{h_randShift[k]}\t");
                            if (k % 3 == 2)
                            {
                                Console.WriteLine("");
                            }
                        }
                        Console.WriteLine("");

                    }
                    projected.Dispose();
                    error.Dispose();
                    error_sqr.Dispose();
                    loss.Dispose();

                }

            }

            //Test volume rastering
            if(false)
            {
                TorchTensor Random_ProjectionOrientations = Float32Tensor.Zeros(new long[] { numOrientations, 3, 3 }, DeviceType.CUDA, 0, true);
                {
                    float[] h_data = Helper.ArrayOfFunction(i =>
                    {

                        Matrix3 rotation = Matrix3.Euler((float)(20 * Math.PI / 180), 0, 0);
                        return rotation.ToArray()[i % 9];
                    }, 9 * numOrientations);
                    GPU.CopyHostToDevice(h_data, Random_ProjectionOrientations.DataPtr(), 9 * numOrientations);
                    GPU.CheckGPUExceptions();
                }
                Random_ProjectionOrientations = Random_ProjectionOrientations.Detach().RequiresGrad(true);
                TorchTensor GT_volumes = proj.RasterToCartesian(atomPositions, projectionOrientations, Float32Tensor.Zeros(new long[] { numOrientations, 3 }));
                GPU.CheckGPUExceptions();
                Optimizer optim = Optimizer.SGD(new List<TorchTensor> { Random_ProjectionOrientations }, 0.1);

                int numIt = 100;
                if (false)
                {
                    Image GT_VolumesImage = new(new int3(dim, dim, dim));
                    GPU.CopyDeviceToDevice(GT_volumes.DataPtr(), GT_VolumesImage.GetDevice(Intent.Write), GT_VolumesImage.ElementsReal);
                    GT_VolumesImage.WriteMRC($"GT_VolumesImage.mrc", true);


                }

                for (int i = 0; i < numIt; i++)
                {
                    optim.ZeroGrad();
                    TorchTensor theseAtomPositions = atomPositions.Detach().RequiresGrad(true);
                    TorchTensor Pred_Volumes = proj.RasterToCartesian(theseAtomPositions, Random_ProjectionOrientations, Float32Tensor.Zeros(new long[] { numOrientations, 3}));
                    if (false)
                    {
                        Image Pred_VolumesImage = new(new int3(dim, dim, dim));
                        GPU.CopyDeviceToDevice(Pred_Volumes.DataPtr(), Pred_VolumesImage.GetDevice(Intent.Write), Pred_VolumesImage.ElementsReal);
                        Pred_VolumesImage.WriteMRC($"Pred_VolumesImage.mrc", true);


                    }

                    TorchTensor error = Pred_Volumes - GT_volumes.Detach();
                    TorchTensor errorSqrd = error.Pow(2);
                    TorchTensor loss = dim*numOrientations*errorSqrd.Mean();
                    Console.WriteLine($"Loss:{loss.Detach().ToDevice(DeviceType.CPU).ToDouble()}");
                    loss.Backward();

                    if (false)
                    {
                        float[] h_neworientations = new float[9 * numOrientations];
                        GPU.CopyDeviceToHost(Random_ProjectionOrientations.DataPtr(), h_neworientations, 9 * numOrientations);
                        for (int j = 0; j < numOrientations; j++)
                        {
                            Console.WriteLine($"OldMatrix {i}.{j} :\n{h_neworientations[9 * j + 0]}\t{h_neworientations[9 * j + 1]}\t{h_neworientations[9 * j + 2]}\n{h_neworientations[9 * j + 3]}\t{h_neworientations[9 * j + 4]}\t{h_neworientations[9 * j + 5]}\n{h_neworientations[9 * j + 6]}\t{h_neworientations[9 * j + 7]}\t{h_neworientations[9 * j + 8]}");
                        }
                    }
                    if (false)
                    {
                        float[] h_neworientations = new float[9 * numOrientations];
                        GPU.CopyDeviceToHost(Random_ProjectionOrientations.Grad().DataPtr(), h_neworientations, 9 * numOrientations);
                        for (int j = 0; j < numOrientations; j++)
                        {
                            Console.WriteLine($"Gradient {i}.{j} :\n{h_neworientations[9 * j + 0]}\t{h_neworientations[9 * j + 1]}\t{h_neworientations[9 * j + 2]}\n{h_neworientations[9 * j + 3]}\t{h_neworientations[9 * j + 4]}\t{h_neworientations[9 * j + 5]}\n{h_neworientations[9 * j + 6]}\t{h_neworientations[9 * j + 7]}\t{h_neworientations[9 * j + 8]}");
                        }

                    }


                    optim.Step();

                    Pred_Volumes.Dispose();
                    error.Dispose();
                    errorSqrd.Dispose();
                    loss.Dispose();

                }

            }



            
        }

        static void TestGridSampleAndProject()
        {
            int processingDevice = 1;
            GPU.SetDevice(processingDevice);
            Image sourceVolumeOrg = Image.FromFile(@"D:\EMD\9233\emd_9233_2.0.mrc");


            float3[] AllAngles = Helper.GetHealpixAngles(3).Select(s => s * Helper.ToRad).ToArray();

            //Warp based Fourier Cropping for reference
            Image sourceVolume = sourceVolumeOrg.AsScaled(new int3(64));
            sourceVolume.WriteMRC("WarpCropped.mrc", true);

            // Create a cropped volume using Torch Operator
            TorchTensor TensorFullVolume = Float32Tensor.Random(new long[] { 1, 1, sourceVolumeOrg.Dims.Z, sourceVolumeOrg.Dims.Y, sourceVolumeOrg.Dims.X }, DeviceType.CUDA, processingDevice);
            GPU.CopyDeviceToDevice(sourceVolumeOrg.GetDevice(Intent.Read), TensorFullVolume.DataPtr(), sourceVolumeOrg.ElementsReal);

            TorchTensor TensorOwnScaledVolume = ScaleVolume(TensorFullVolume, 3, 64, 64, 64);
            Image imageOwnScaledVolume = new Image(new int3(64));
            GPU.CopyDeviceToDevice(TensorOwnScaledVolume.DataPtr(), imageOwnScaledVolume.GetDevice(Intent.Write), imageOwnScaledVolume.ElementsReal);
            imageOwnScaledVolume.WriteMRC("imageOwnScaledVolume.mrc", true);

            TorchTensor TensorWarpScaledVolume = Float32Tensor.Random(new long[] { 1, 1, sourceVolume.Dims.Z, sourceVolume.Dims.Y, sourceVolume.Dims.X }, DeviceType.CUDA, processingDevice);
            GPU.CopyDeviceToDevice(sourceVolume.GetDevice(Intent.Read), TensorWarpScaledVolume.DataPtr(), sourceVolume.ElementsReal);

            ReconstructionWGANGenerator RefGen = ReconstructionWGANGenerator(TensorOwnScaledVolume, 64);
            RefGen.ToCuda(processingDevice);

            TorchTensor EmptyVol = Float32Tensor.Zeros(TensorOwnScaledVolume.Shape, DeviceType.CUDA, processingDevice);
            ReconstructionWGANGenerator TargetGen = ReconstructionWGANGenerator(EmptyVol, 64);
            TargetGen.ToCuda(processingDevice);

            //Set stuff needed for a quick reconstruction
            int batchSize = 32;
            TorchTensor tensorAngles = Float32Tensor.Empty(new long[] { batchSize, 3 }, DeviceType.CUDA, 1);
            Optimizer optim = Optimizer.Adam(TargetGen.GetParameters(), 0.1, 1e-4);
            Random ReloadRand = new Random(42);
            for (int i = 0; i < 1000; i++)
            {
                optim.ZeroGrad();
                int[] AngleIds = Helper.ArrayOfFunction(i => ReloadRand.Next(0, AllAngles.Length), batchSize);

                // Read, and copy or rescale real and fake images from prepared stacks

                float3[] theseAngles = Helper.IndexedSubset(AllAngles, AngleIds);
                GPU.CopyHostToDevice(Helper.ToInterleaved(theseAngles), tensorAngles.DataPtr(), batchSize * 3);
                using (TorchTensor RefProj = RefGen.Forward(tensorAngles, false))
                using (TorchTensor TargetProj = TargetGen.ForwardNew(tensorAngles, false))
                using (TorchTensor Loss = (RefProj - TargetProj).Pow(2).Sum().Mean())
                {
                    Loss.Backward();
                    optim.Step();
                    if (i % 100 == 0 || i == 1000 - 1)
                    {
                        Image imageRefProj = new Image(new int3(64, 64, batchSize));
                        GPU.CopyDeviceToDevice(RefProj.DataPtr(), imageRefProj.GetDevice(Intent.Write), imageRefProj.ElementsReal);
                        imageRefProj.WriteMRC($"imageRefProj_{i}.mrc", true);
                        Image imageTargetProj = new Image(new int3(64, 64, batchSize));
                        GPU.CopyDeviceToDevice(TargetProj.DataPtr(), imageTargetProj.GetDevice(Intent.Write), imageTargetProj.ElementsReal);
                        imageTargetProj.WriteMRC($"imageTargetProj_{i}.mrc", true);
                    }
                }
            }
        }


        private static float getGaussian(Random rand, double mu, double sigma)
        {
            double u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            return (float)(mu + sigma * randStdNormal);
        }

        private static void TestReconstructionFromRealProjections()
        {
            int seed = 42;
            int device = 0;
            //int DimGenerator = 64;
            int DimGenerator = 128;
            int DimZoom = 128;
            int BatchSize = 512*16;
            double sigmaShiftPix = 2.0;
            string WorkingDirectory = @"D:\GAN_recon_polcompl\";
            string OutDirectory = @"D:\GAN_recon_polcompl\reconstructionTestExperimental";
            string outFileName = $@"{OutDirectory}\fscValues.txt";
            if (!Directory.Exists(OutDirectory))
            {
                Directory.CreateDirectory(OutDirectory);
            }
            using (var dump = new StreamWriter(outFileName, false))
            {
            }


            GPU.SetDevice(device);

            Star TableIn = new Star(Path.Combine(WorkingDirectory, "run_data.star"), "particles");
            TableIn.AddColumn("rlnVoltage", "200.0");
            TableIn.AddColumn("rlnSphericalAberration", "2.7");
            TableIn.AddColumn("rlnAmplitudeContrast", "0.07");
            TableIn.AddColumn("rlnDetectorPixelSize", "3.0");
            TableIn.AddColumn("rlnMagnification", "10000");

            Random rand = new Random(seed);

            string[] ColumnStackNames = TableIn.GetColumn("rlnImageName").Select(s => Helper.PathToNameWithExtension(s.Substring(s.IndexOf('@') + 1))).ToArray();
            HashSet<string> UniqueStackNames = Helper.GetUniqueElements(ColumnStackNames);


            int DimRaw = MapHeader.ReadFromFile(Path.Combine(WorkingDirectory, "Refine3D_CryoSparcSelected_run_class001.mrc")).Dimensions.X;

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
            int[] AllIDs = Helper.ArrayOfSequence(0, AllParticleAddresses.Length, 1);

            CTF[] AllParticleCTF = TableIn.GetRelionCTF();
            float3[] AllParticleAngles = TableIn.GetRelionAngles().Select(s => s * Helper.ToRad).ToArray();
            float3[] AllParticleAnglesDeg = TableIn.GetRelionAngles();
            float3[] AllParticleShifts = TableIn.GetRelionOffsets().Select(s => s / 3.0f).ToArray();

            Image TrefVolume = Image.FromFile(Path.Combine(WorkingDirectory, "Refine3D_CryoSparcSelected_run_class001.mrc"));
            if (DimZoom != DimRaw)
                TrefVolume = TrefVolume.AsRegion(new int3((DimRaw - DimZoom) / 2), new int3(DimZoom));
            Image RefVolumeScaled;
            if (DimGenerator < DimZoom)
                RefVolumeScaled = TrefVolume.AsScaled(new int3(DimGenerator));
            else
                RefVolumeScaled = TrefVolume;


            Random ReloadRand = new Random(seed);


            Image ParticleStack = new Image(new int3(DimRaw, DimRaw, BatchSize));

            Image CTFFull = new Image(new int3(DimZoom, DimZoom, BatchSize), true);
            CTFFull.Fill(1);
            Image CTFScaled = new Image(new int3(DimGenerator, DimGenerator, BatchSize), true);
            CTFScaled.Fill(1);

            Image CTFCoordsFull = CTF.GetCTFCoords(DimZoom, DimZoom);
            Image CTFCoordsScaled = CTF.GetCTFCoords(DimGenerator, DimZoom);
            float2[][] scaledCoords = CTFCoordsScaled.GetHostComplexCopy();
            float2[][] FullCoords = CTFCoordsFull.GetHostComplexCopy();



            {
                // If this thread succeeded at pushing its previously loaded batch to processing
                Projector Reconstructor = new(new int3(DimGenerator), 2);
                for (int iter = 0; (iter+1)*BatchSize < AllParticleAngles.Length; iter++)
                {
                    int[] SubsetIDs = Helper.RandomSubset(AllIDs, BatchSize, ReloadRand.Next());
                    int[] currentSubset = Helper.ArrayOfSequence(iter * BatchSize, (iter + 1) * BatchSize, 1);

                    float3[] theseAngles = Helper.IndexedSubset(AllParticleAngles, SubsetIDs);
                    float3[] theseShifts = Helper.IndexedSubset(AllParticleShifts, SubsetIDs);

                    // Read, and copy or rescale real and fake images from prepared stacks
                    float[][] ParticleStackData = ParticleStack.GetHost(Intent.Write);
                    //Helper.ForCPU(0, BatchSize, 3, null, (b,threadID) =>
                    for (int b = 0; b < BatchSize; b++)
                    {
                        int id = SubsetIDs[b];
                        IOHelper.ReadMapFloat(Path.Combine(WorkingDirectory, "particles", AllParticleAddresses[id].name),
                                                new int2(1),
                                                0,
                                                typeof(float),
                                                new[] { AllParticleAddresses[id].id },
                                                null,
                                                new[] { ParticleStackData[b] });
                    }//, null);

                    GPU.Normalize(ParticleStack.GetDevice(Intent.Read),
                        ParticleStack.GetDevice(Intent.Write),
                        (uint)ParticleStack.ElementsSliceReal,
                        (uint)BatchSize);
                    ParticleStack.ShiftSlices(theseShifts);

                    var theseCTF = Helper.IndexedSubset(AllParticleCTF, SubsetIDs).Select(c => c.ToStruct()).ToArray();
                    GPU.CreateCTF(CTFFull.GetDevice(Intent.Write),
                                  CTFCoordsFull.GetDevice(Intent.Read),
                                  IntPtr.Zero,
                                  (uint)CTFCoordsFull.ElementsSliceComplex,
                                  theseCTF,
                                  false,
                                  (uint)BatchSize);
                    GPU.CreateCTF(CTFScaled.GetDevice(Intent.Write),
                      CTFCoordsScaled.GetDevice(Intent.Read),
                      IntPtr.Zero,
                      (uint)CTFCoordsScaled.ElementsSliceComplex,
                      theseCTF,
                      false,
                      (uint)BatchSize);


                    {
                        Image fft = ParticleStack.AsFFT();
                        ParticleStack.Dispose();
                        fft.Multiply(CTFFull);
                        ParticleStack = fft.AsIFFT(false, 0, true);
                        fft.Dispose();
                    }
                    CTFScaled.Multiply(CTFScaled);
                    CTFFull.Multiply(CTFFull);

                    //CTFScaled.WriteMRC(@$"{OutDirectory}\CTFScaled_{iter}.mrc", true);

                    Image projectedScaled;
                    if (DimGenerator < DimZoom)
                        projectedScaled = ParticleStack.AsScaled(new int2(DimGenerator));
                    else
                        projectedScaled = ParticleStack.GetCopy();

                    //projectedScaled.WriteMRC(@$"{OutDirectory}\projectedScaled_{iter}.mrc", true);

                    Image projectedScaledFFT = projectedScaled.AsFFT(false);
                    projectedScaledFFT.ShiftSlices(Helper.ArrayOfFunction(i => new float3(projectedScaled.Dims.X / 2, projectedScaled.Dims.Y / 2, 0), BatchSize));
                    projectedScaled.Dispose();


                    Reconstructor.BackProject(projectedScaledFFT, CTFScaled, theseAngles, new float3(1, 1, 0));
                    projectedScaledFFT.Dispose();

                    Image reconstruction = Reconstructor.Reconstruct(false);
                    reconstruction.WriteMRC($@"{OutDirectory}\Reconstruction_{iter}.mrc", true);

                    
                    float[] fsc = FSC.GetFSC(reconstruction, RefVolumeScaled);
                    using (StreamWriter w = File.AppendText(outFileName))
                    {
                        for (int i = 0; i < fsc.Length; i++)
                        {
                            w.WriteLine($"{iter}\t{DimGenerator * (3.0) / (i + 1)}\t{fsc[i]}");
                        }

                    }
                    reconstruction.Dispose();

                }
            }
        }



        private static void TestReconstructionFromArtificialProjections()
        {
            int seed = 42;
            int device = 0;
            int DimGenerator = 64;
            int DimZoom = 128;
            int BatchSize = 512;
            double sigmaShiftPix = 2.0;
            string WorkingDirectory = @"D:\GAN_recon_polcompl\";
            string OutDirectory = @"D:\GAN_recon_polcompl\reconstructionTest";
            string outFileName = $@"{OutDirectory}\fscValues.txt";
            if (!Directory.Exists(OutDirectory))
            {
                Directory.CreateDirectory(OutDirectory);
            }
            using (var dump = File.OpenWrite(outFileName))
            { 
            }


            Torch.SetSeed(seed);

            GPU.SetDevice(device);

            Star TableIn = new Star(Path.Combine(WorkingDirectory, "cryosparc_P243_J525_003_particles.star"));

            Random rand = new Random(seed);

            string[] ColumnStackNames = TableIn.GetColumn("rlnImageName").Select(s => Helper.PathToNameWithExtension(s.Substring(s.IndexOf('@') + 1))).ToArray();
            HashSet<string> UniqueStackNames = Helper.GetUniqueElements(ColumnStackNames);


            int DimRaw = MapHeader.ReadFromFile(Path.Combine(WorkingDirectory, "Refine3D_CryoSparcSelected_run_class001.mrc")).Dimensions.X;

            var AllParticleAddresses = new (int id, string name)[TableIn.RowCount];

            int[] AllIDs = Helper.ArrayOfSequence(0, AllParticleAddresses.Length, 1);

            CTF[] AllParticleCTF = TableIn.GetRelionCTF();
            float3[] AllParticleAngles = TableIn.GetRelionAngles().Select(s => s * Helper.ToRad).ToArray();

            float3[] RandomParticleAngles = Helper.GetHealpixAngles(3).Select(s => s * Helper.ToRad).ToArray();

            int numParticles = RandomParticleAngles.Length;

            string starFileName = @$"{WorkingDirectory}\run_model.star";
            float[][] AllSigmas = Helper.ArrayOfFunction(i =>
            {
                Star table = new(starFileName, $"model_group_{i + 1}");
                string[] column = table.GetColumn("rlnSigma2Noise");
                float[] entries = column.Select(s => (float)Math.Sqrt(float.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture))).ToArray();
                return entries;
            }, 1);

            Image cleanProjection;
            {
                Image RefVolume = Image.FromFile(Path.Combine(WorkingDirectory, "Refine3D_CryoSparcSelected_run_class001.mrc"));
                if (DimZoom != DimRaw)
                    RefVolume = RefVolume.AsRegion(new int3((DimRaw - DimZoom) / 2), new int3(DimZoom));
                Projector CleanProj = new Projector(RefVolume, 2);
                Image projected = CleanProj.ProjectToRealspace(new int2(DimZoom), new float3[] { new float3(0) });
                cleanProjection = projected.AsScaled(new int2(DimGenerator));
                cleanProjection.Normalize();
                cleanProjection.MaskSpherically(DimGenerator / 2, DimGenerator / 8, false);

                cleanProjection.FreeDevice();
                CleanProj.Dispose();

            }

            Image TrefVolume = Image.FromFile(Path.Combine(WorkingDirectory, "Refine3D_CryoSparcSelected_run_class001.mrc"));
            if (DimZoom != DimRaw)
                TrefVolume = TrefVolume.AsRegion(new int3((DimRaw - DimZoom) / 2), new int3(DimZoom));
            Image RefVolumeScaled = TrefVolume.AsScaled(new int3(DimGenerator));
            Projector TProj = new Projector(TrefVolume, 2);

            Random ReloadRand = new Random(seed);
            RandomNormal NoiseRand = new RandomNormal(seed);
            Random ShiftRand = new Random(seed);
            Random sigmaPicker = new Random(seed);

            Image LoadStack = new Image(new int3(DimRaw, DimRaw, BatchSize));

            Image TImagesReal = new Image(new int3(DimGenerator, DimGenerator, BatchSize));
            Image TImagesNoise = new Image(new int3(DimGenerator, DimGenerator, BatchSize));
            Image CTFFull = new Image(new int3(DimZoom, DimZoom, BatchSize), true);
            CTFFull.Fill(1);
            Image CTFScaled = new Image(new int3(DimGenerator, DimGenerator, BatchSize), true);
            CTFScaled.Fill(1);

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

            {
                // If this thread succeeded at pushing its previously loaded batch to processing
                Projector Reconstructor = new(new int3(DimGenerator), 2);
                for (int iter = 0; iter < 100; iter++)
                {
                    int[] SubsetIDs = Helper.RandomSubset(AllIDs, BatchSize, ReloadRand.Next());
                    int[] AngleIds = Helper.ArrayOfFunction(i => ReloadRand.Next(0, RandomParticleAngles.Length), BatchSize);
                    int[] SigmaIds = Helper.ArrayOfFunction(i => sigmaPicker.Next(0, AllSigmas.Length), BatchSize);

                    // Read, and copy or rescale real and fake images from prepared stacks

                    float3[] theseAngles = Helper.IndexedSubset(RandomParticleAngles, AngleIds);
                    //theseAngles = Helper.ArrayOfFunction(i => new float3(0, 0, 0), AngleIds.Length);
                    float[][] theseSigmas = Helper.IndexedSubset(AllSigmas, SigmaIds);

                    Image projected = TProj.ProjectToRealspace(new int2(DimZoom), theseAngles);

                    GPU.CheckGPUExceptions();
                    var theseCTF = Helper.IndexedSubset(AllParticleCTF, SubsetIDs).Select(c => c.ToStruct()).ToArray();
                    GPU.CreateCTF(CTFFull.GetDevice(Intent.Write),
                                  CTFCoordsFull.GetDevice(Intent.Read),
                                  IntPtr.Zero,
                                  (uint)CTFCoordsFull.ElementsSliceComplex,
                                  theseCTF,
                                  false,
                                  (uint)BatchSize);
                    GPU.CreateCTF(CTFScaled.GetDevice(Intent.Write),
                      CTFCoordsScaled.GetDevice(Intent.Read),
                      IntPtr.Zero,
                      (uint)CTFCoordsScaled.ElementsSliceComplex,
                      theseCTF,
                      false,
                      (uint)BatchSize);

                    {
                        Image thisCTFSign = CTFFull.GetCopy();
                        thisCTFSign.Sign();
                        CTFFull.Multiply(thisCTFSign);
                        CTFFull.Multiply(CTFMaskFull);
                        thisCTFSign.Dispose();
                    }
                    {
                        Image thisCTFSign = CTFScaled.GetCopy();
                        thisCTFSign.Sign();
                        CTFScaled.Multiply(thisCTFSign);
                        CTFScaled.Multiply(CTFMaskScaled);
                        thisCTFSign.Dispose();
                    }

                    CTFFull.Fill(1);
                    CTFScaled.Fill(1);
                    {
                        Image fft = projected.AsFFT();
                        projected.Dispose();
                        fft.Multiply(CTFFull);
                        projected = fft.AsIFFT(false, 0, true);
                        fft.Dispose();
                    }

                    Image ColoredNoiseFFT = new Image(projected.Dims, true, true);
                    float2[][] complexData = Helper.ArrayOfFunction(i => new float2[projected.DimsFTSlice.Elements()], projected.Dims.Z);
                    for (int z = 0; z < projected.Dims.Z; z++)
                    {
                        for (int y = 0; y < projected.Dims.Y; y++)
                        {
                            for (int x = 0; x < projected.Dims.X / 2 + 1; x++)
                            {
                                float yy = y >= projected.Dims.Y / 2 + 1 ? y - projected.Dims.Y : y;
                                yy *= yy;

                                float xx = x;
                                xx *= xx;

                                float r = (float)Math.Sqrt(xx + yy);
                                int waveNumber = (int)Math.Round(r);
                                waveNumber = Math.Min(waveNumber, projected.Dims.X / 2);
                                complexData[z][y * (projected.Dims.X / 2 + 1) + x] = new float2(NoiseRand.NextSingle(0, theseSigmas[z][waveNumber]), NoiseRand.NextSingle(0, theseSigmas[z][waveNumber]));

                            }
                        }
                    }
                    ColoredNoiseFFT.UpdateHostWithComplex(complexData);
                    Image ColoredNoise = ColoredNoiseFFT.AsIFFT();
                    ColoredNoise.Normalize();
                    Image NoiseScaled = ColoredNoise.AsScaled(new int2(DimGenerator));
                    GPU.CopyDeviceToDevice(NoiseScaled.GetDevice(Intent.Read), TImagesNoise.GetDevice(Intent.Write), NoiseScaled.ElementsReal);
                    NoiseScaled.Dispose();
                    ColoredNoiseFFT.Dispose();
                    projected.Add(ColoredNoise);
                    ColoredNoise.Dispose();

                    Image projectedScaled = projected.AsScaled(new int2(DimGenerator));
                    projectedScaled.Bandpass(0, 1.0f, false, 0.05f);
                    projected.Dispose();
                    Image projectedScaledFFT = projectedScaled.AsFFT(false);
                    
                    projectedScaledFFT.ShiftSlices(Helper.ArrayOfFunction(i => new float3(projectedScaled.Dims.X / 2, projectedScaled.Dims.Y / 2, 0), BatchSize));
                    projectedScaled.WriteMRC(@$"{OutDirectory}\projectedScaled_{iter}.mrc", true);
                    projectedScaled.Dispose();
                    Reconstructor.BackProject(projectedScaledFFT, CTFScaled, theseAngles, new float3(1, 1, 0));
                    Image reconstruction = Reconstructor.Reconstruct(false);
                    reconstruction.WriteMRC($@"{OutDirectory}\Reconstruction_{iter}.mrc", true);
                    
                    projectedScaledFFT.Dispose();
                    float[] fsc = FSC.GetFSC(reconstruction, RefVolumeScaled);
                    using (StreamWriter w = File.AppendText(outFileName))
                    {
                        for (int i = 0; i < fsc.Length; i++)
                        {
                            w.WriteLine($"{iter}\t{DimGenerator * (6) / (i+1)}\t{fsc[i]}");
                        }
                        
                    }
                    reconstruction.Dispose();

                }
            }
        }
        
           
        static void Main(string[] args)
        {
            /*
            Image im = Image.FromFile(@"Z:\DATA\fjochhe\6yyt_molmap\6yyt_map_3A.mrc");
            im = im.AsPadded(new int3(256));
            im.WriteMRC(@"Z:\DATA\fjochhe\6yyt_molmap\6yyt_map_3A_new.mrc",true);
            im.AsScaled(new int3(128)).WriteMRC(@"Z:\DATA\fjochhe\6yyt_molmap\6yyt_map_6A.mrc", true);
            */
            TestReconstructionFromRealProjections();
            //TestGridSampleAndProject();
        }
    }
}
