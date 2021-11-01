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

namespace TildeDataLoader
{
    class TildeDataLoader
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

        static void Main(string[] args)
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

            //Test FFT Crop speed
            if(true)
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
                        TorchTensor newCroppedFFT = FFTCrop(thisFullFFT, 3,  50, 50, 50);
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

            //Tesst FFT Crop vs warp crop
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
                TorchTensor TensorFFTCroppedVolume = FFTCrop(TensorFFTFullVolume, 3,cropDim, cropDim, cropDim);
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
    }
}
