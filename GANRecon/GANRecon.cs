
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp.Tensor;
using Warp;
using Warp.NNModels;
using Warp.Tools;

namespace GANRecon
{
    class GANRecon
    {


        static void Main(string[] args)
        {
            int boxLength = 33;
            int2 boxsize = new(boxLength);
            int[] devices = { 0, 1 };
            int batchSize = 8;
            int numEpochs = 100;
            //var NoiseNet = new NoiseNet2DTorch(boxsize, devices, batchSize);

            //Read all particles and CTF information into memory
            {
                var directory = @"D:\GANRecon";
                var refVolume = Image.FromFile(@"D:\GANRecon\run_1k_unfil.mrc").AsScaled(new int3(boxLength));

                Star particles = new Star($@"{directory}\run_data.star");
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
                    Image im = Image.FromFile($@"{directory}\{uniquePaths[i]}");
                    var scaled = im.AsScaled(boxsize);
                    scaled.FreeDevice();
                    stacksByPath[uniquePaths[i]] = scaled;
                    im.Dispose();
                    //}
                }

                
                

                int oversampling = 1;
                int Oversampled = boxLength;// 2 * (oversampling * (boxLength / 2) + 1) + 1;
                int3 DimsOversampled = new int3(Oversampled, Oversampled, Oversampled);

                float[] coordinates = new float[boxLength * boxLength * 3];
                for (int y = 0; y < boxLength; y++)
                {
                    for (int x = 0; x < boxLength/2+1; x++)
                    {
                        coordinates[y * boxLength*3 + x * 3 + 0] = ((float)x) / ((float)oversampling*boxLength - 1) * 2 - 1;
                        coordinates[y * boxLength*3 + x * 3 + 1] = ((float)y) / ((float)oversampling * boxLength - 1) * 2 - 1;
                        coordinates[y * boxLength*3 + x * 3 + 2] = 0;
                    }
                }
                TorchTensor tensorCoordinates = Float32Tensor.Zeros(new long[] { 1, 1, boxLength, boxLength/2+1, 3 }, TorchSharp.DeviceType.CUDA);

                GPU.CopyHostToDevice(coordinates, tensorCoordinates.DataPtr(), coordinates.Length);
                GPU.CheckGPUExceptions();

                Image correctGriddingVolume = refVolume.GetCopy();
                correctGriddingVolume.TransformValues((x, y, z, val) =>
                {
                    float xx = x - (float)refVolume.Dims.X / 2;
                    float yy = y - (float)refVolume.Dims.Y / 2;
                    float zz = z - (float)refVolume.Dims.Z / 2;
                    float r = (float)Math.Sqrt((float)(xx * xx + yy * yy + zz * zz));
                    if (r > 0)
                    {
                        float rval = r / (refVolume.Dims.X * oversampling);
                        float sinc = (float)Math.Sin(Math.PI * rval) / (float)(Math.PI * rval);

                        return val / (sinc * sinc);
                    }
                    else
                        return val;
                });
                correctGriddingVolume.WriteMRC(@$"{directory}\correctGriddingVolume.mrc", true);
                TorchTensor volume = Float32Tensor.Zeros(new long[] { 1, 1, boxLength, boxLength, boxLength }, TorchSharp.DeviceType.CUDA);
                GPU.CopyDeviceToDevice(correctGriddingVolume.GetDevice(Intent.Read), volume.DataPtr(), refVolume.ElementsReal);
                
                TorchTensor volumePadded = oversampling>1?volume.Pad(new long[] { ((oversampling-1)* boxLength) / 2,((oversampling-1)* boxLength) / 2,
                    ((oversampling-1)* boxLength) / 2, ((oversampling-1)* boxLength) / 2,
                    ((oversampling-1)* boxLength) / 2,((oversampling-1)* boxLength) / 2}):volume;
                double max_r2 = Math.Pow(Math.Min((float)Oversampled / 2, (float)boxLength / 2), 2)* oversampling* oversampling;
                TorchTensor fftVolume = volume.fftn(new long[] { 2, 3, 4 });
                GPU.CheckGPUExceptions();
                TorchTensor fftShiftedVolume = fftVolume.fftshift(new long[] { 2, 3, 4 });
                GPU.CheckGPUExceptions();
                TorchTensor fftShiftedSlice = TorchSharp.NN.Modules.GridSample(fftShiftedVolume.ToDevice(TorchSharp.DeviceType.CPU),
                    tensorCoordinates.ToDevice(TorchSharp.DeviceType.CPU), max_r2
                    ).ToDevice(TorchSharp.DeviceType.CUDA);
                //GPU.CheckGPUExceptions();
                TorchTensor fftSlice = fftShiftedSlice.ifftshift(new long[] { 2, 3, 4 });
                GPU.CheckGPUExceptions();
                TorchTensor slice = fftSlice.ifftn(new long[] { 3, 4 });
                GPU.CheckGPUExceptions();
                {
                    Image imagefftVolume = new Image(new int3(boxLength, boxLength, boxLength));
                    GPU.CopyDeviceToDevice(fftVolume.Abs().DataPtr(), imagefftVolume.GetDevice(Intent.Write), imagefftVolume.ElementsReal);
                    imagefftVolume.WriteMRC(@$"{directory}\imagefftVolume.mrc", true);

                    Image imagefftShiftedVolume = new Image(new int3(boxLength, boxLength, boxLength));
                    GPU.CopyDeviceToDevice(fftShiftedVolume.Abs().DataPtr(), imagefftShiftedVolume.GetDevice(Intent.Write), imagefftShiftedVolume.ElementsReal);
                    imagefftShiftedVolume.WriteMRC(@$"{directory}\imagefftShiftedVolume.mrc", true);

                    Image imagefftShiftedSlice = new Image(new int3(boxLength, boxLength, 1));
                    GPU.CopyDeviceToDevice(fftShiftedSlice.Abs().DataPtr(), imagefftShiftedSlice.GetDevice(Intent.Write), imagefftShiftedSlice.ElementsReal); ;
                    imagefftShiftedSlice.WriteMRC(@$"{directory}\imagefftShiftedSlice.mrc", true);

                    Image imagefftSlice = new Image(new int3(boxLength, boxLength, 1));
                    GPU.CopyDeviceToDevice(fftSlice.Abs().DataPtr(), imagefftSlice.GetDevice(Intent.Write), imagefftSlice.ElementsReal); ;
                    imagefftSlice.WriteMRC(@$"{directory}\imagefftSlice.mrc", true);

                    Image imageSlice = new Image(new int3(boxLength, boxLength, 1));
                    GPU.CopyDeviceToDevice(slice.Abs().Clone().DataPtr(), imageSlice.GetDevice(Intent.Write), imageSlice.ElementsReal);
                    imageSlice.WriteMRC(@$"{directory}\imageSlice.mrc", true);

                }
                /*
                Projector refProjector = new(refVolume, 1);
                refProjector.ProjectToRealspace(new int2(boxLength), new float3[] { new float3(0) }).WriteMRC(@$"{directory}\RealspaceProject.mrc", true);
                Image fftProject = refProjector.Project(new int2(boxLength), new float3[] { new float3(0) }).AsAmplitudes();
                fftProject.WriteMRC(@$"{directory}\FFTProject.mrc");*/
                return;
                
                /*
                List<float> losses = new();
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

                    Projector proj = new Projector(refVolume, 1);
                    Image CTFCoords = CTF.GetCTFCoords(boxLength*2, boxLength * 2);
                    Random rnd = new Random();
                    for (int epoch = 0; epoch < numEpochs; epoch++)
                    {
                        float meanLoss = 0.0f;
                        for (int numBatch = 0; numBatch < count/batchSize; numBatch++)
                        {

                            int[] thisBatch = Helper.ArrayOfFunction(i => rnd.Next(count), batchSize);
                            Image BatchParticles = Image.Stack(Helper.ArrayOfFunction(i=>SubsetParticles[thisBatch[i]],batchSize));
                            BatchParticles.ShiftSlices(Helper.ArrayOfFunction(i => SubsetOffsets[thisBatch[i]], batchSize));
                            var BatchCTFStructs = Helper.ArrayOfFunction(i => SubsetCTFs[thisBatch[i]], batchSize);

                            Image BatchCTFs = new Image(new int3(boxLength * 2, boxLength * 2, batchSize), true);
                            GPU.CreateCTF(BatchCTFs.GetDevice(Intent.Write),
                                            CTFCoords.GetDevice(Intent.Read),
                                            IntPtr.Zero,
                                            (uint)CTFCoords.ElementsSliceComplex,
                                            BatchCTFStructs.Select(p => p.ToStruct()).ToArray(),
                                            false,
                                            (uint)BatchParticles.Dims.Z);
                            GPU.CheckGPUExceptions();
                            var BatchAngles = Helper.ArrayOfFunction(i => SubsetAngles[thisBatch[i]], batchSize);
                            Image source = proj.ProjectToRealspace(boxsize, BatchAngles);
                            source.Normalize();
                            BatchParticles.Normalize();
                            NoiseNet.Train(source, BatchParticles, BatchCTFs, 0.01f, out Image prediction, out Image predictionDeconv, out float[] loss);
                            meanLoss += loss[0];

                            BatchParticles.Dispose();
                            BatchCTFs.Dispose();
                            source.Dispose();
                        }
                        losses.Append(meanLoss / (count / batchSize));
                        Console.WriteLine($"Epoch {epoch}: {meanLoss / (count / batchSize)}");
                        GPU.CheckGPUExceptions();
                        if (epoch % 10 == 0)
                        {
                            NoiseNet.Save($@"models\NoiseNet_e{epoch}.model");
                        }
                    }
                    NoiseNet.Save(@"models\NoiseNet.model");


                }
                */
            }
        }
    }
}
