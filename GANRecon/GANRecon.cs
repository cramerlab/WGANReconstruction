
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
            int boxLength = 34;
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
                int Oversampled = oversampling * (boxLength);
                double max_r2 = Math.Pow(Math.Min((float)Oversampled / 2, (float)boxLength / 2), 2) * oversampling * oversampling;
                int3 DimsOversampled = new int3(Oversampled, Oversampled, Oversampled);

                float[] coordinates = new float[boxLength * (boxLength/2+1) * 3];

                for (int y = 0; y < boxLength; y++)
                {
                    for (int x = 0; x < boxLength / 2 + 1; x++)
                    {
                        float xx = x;
                        float yy = y < boxLength / 2  ? y + (boxLength/2 ) : y  - boxLength/2;

                        coordinates[y * (boxLength / 2 + 1) * 3 + x * 3 + 0] = ((float)xx) / ((float)(((int)boxLength / 2)+1) - 1) * 2 - 1 ;
                        coordinates[y * (boxLength / 2 + 1) * 3 + x * 3 + 1] = ((float)yy) / ((float)boxLength - 1) * 2 - 1;
                        coordinates[y * (boxLength / 2 + 1) * 3 + x * 3 + 2] = 0;
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
                {
                    Image correctGriddingVolumeShifted = correctGriddingVolume.AsShiftedVolume(new float3((float)correctGriddingVolume.Dims.X / 2, (float)correctGriddingVolume.Dims.Y / 2, (float)correctGriddingVolume.Dims.Z / 2));
                    correctGriddingVolumeShifted.WriteMRC(@$"{directory}\correctGriddingVolumeShifted.mrc", true);
                    Image foo_fft = correctGriddingVolumeShifted.AsFFT(true);
                    foo_fft.Multiply(boxLength);
                    //foo_fft = foo_fft.AsPadded(new int3(Oversampled));
                    foo_fft.TransformValues((x, y, z, val) => {

                        int zz = (z < foo_fft.DimsFT.X) ? z : z - foo_fft.DimsFT.Z;
                        int yy = (y < foo_fft.DimsFT.X) ? y : y - foo_fft.DimsFT.Y;
                        int xx = x/2;
                        int r2 = zz * zz + yy * yy + xx * xx;

                        if (r2 <= max_r2)
                        {
                            return val;
                        }
                        return 0.0f;

                    });
                    Image re = foo_fft.AsReal();
                    Image im = foo_fft.AsImaginary();
                    Image.Stack(new Image[] { re, im }).WriteMRC(@$"{directory}\correctGriddingVolumeShiftedWarpFFT.mrc", true);
                    foo_fft.AsIFFT(true, 0, true).WriteMRC(@$"{directory}\correctGriddingVolumeShiftedWarpIFFT.mrc", true);
                }

                {
                    Image correctGriddingVolumeShifted = correctGriddingVolume.AsShiftedVolume(new float3((float)correctGriddingVolume.Dims.X / 2, (float)correctGriddingVolume.Dims.Y / 2, (float)correctGriddingVolume.Dims.Z / 2));
                    correctGriddingVolume.WriteMRC(@$"{directory}\correctGriddingVolume.mrc", true);
                    Image foo_fft = correctGriddingVolumeShifted.AsFFT(true);
                    foo_fft.Multiply(boxLength);
                    //foo_fft = foo_fft.AsPadded(new int3(Oversampled));
                    foo_fft.TransformValues((x, y, z, val) => {

                        int zz = (z < foo_fft.DimsFT.X) ? z : z - foo_fft.DimsFT.Z;
                        int yy = (y < foo_fft.DimsFT.X) ? y : y - foo_fft.DimsFT.Y;
                        int xx = x / 2;
                        int r2 = zz * zz + yy * yy + xx * xx;

                        if (r2 <= max_r2)
                        {
                            return val;
                        }
                        return 0.0f;

                    });
                    Image re = foo_fft.AsReal();
                    Image im = foo_fft.AsImaginary();
                    Image.Stack(new Image[] { re, im }).WriteMRC(@$"{directory}\correctGriddingVolumeWarpFFT.mrc", true);
                    foo_fft.AsAmplitudes().WriteMRC(@$"{directory}\correctGriddingVolumeWarpFFTAbs.mrc", true);
                    correctGriddingVolume = foo_fft.AsIFFT(true, 0, true);
                    correctGriddingVolume.WriteMRC(@$"{directory}\correctGriddingVolumeWarpIFFT.mrc", true);

                }
                Projector refProjector = new(refVolume, 1);

                /*{
                    var path = @$"{directory}\ProjectorData.mrc";

                    Image DataRe = refProjector.Data.AsReal();
                    DataRe.FreeDevice();
                    Image DataIm = refProjector.Data.AsImaginary();
                    DataIm.FreeDevice();

                    Image Combined = Image.Stack(new[] { DataRe, DataIm });
                    DataRe.Dispose();
                    DataIm.Dispose();

                    Combined.WriteMRC(path, true);
                    Combined.Dispose();
                    refProjector.Data.AsIFFT(true).WriteMRC(@$"{directory}\ProjectorDataIFFT.mrc", true);

                }*/
                Image Project = refProjector.ProjectToRealspace(new int2(boxLength), new float3[] { new float3(0) });
                Project.WriteMRC(@$"{directory}\RealspaceProject.mrc", true);
                Image fftProject = Project.AsFFT();
                fftProject = fftProject.AsAmplitudes();
                fftProject.WriteMRC(@$"{directory}\FFTProject.mrc", true);

                TorchTensor volume = Float32Tensor.Zeros(new long[] { 1, 1, boxLength, boxLength, boxLength }, TorchSharp.DeviceType.CUDA);
                GPU.CopyDeviceToDevice(correctGriddingVolume.GetDevice(Intent.Read), volume.DataPtr(), correctGriddingVolume.ElementsReal);
                TorchTensor fftVolume = volume.rfftn(new long[] { 2, 3, 4 });
                //GPU.CopyDeviceToDevice(correctFFT.GetDevice(Intent.Read), fftVolume.DataPtr(), correctGriddingVolume.ElementsReal);
                GPU.CheckGPUExceptions();
                TorchTensor fftShiftedVolume = fftVolume.fftshift(new long[] { 2, 3 });
                GPU.CheckGPUExceptions();
                TorchTensor fftSlice = TorchSharp.NN.Modules.GridSample(fftShiftedVolume.ToDevice(TorchSharp.DeviceType.CPU),
                    tensorCoordinates.ToDevice(TorchSharp.DeviceType.CPU), max_r2
                    ).ToDevice(TorchSharp.DeviceType.CUDA);

                TorchTensor Slice = fftSlice.irfftn(new long[] { 3, 4 });

                {
                    /*{
                        TorchTensor new_volume = Float32Tensor.Zeros(new long[] { 1, 1, boxLength, boxLength, boxLength }, TorchSharp.DeviceType.CUDA);
                        GPU.CopyDeviceToDevice(refVolume.GetDevice(Intent.Read), new_volume.DataPtr(), refVolume.ElementsReal);
                        Image volumeWarpRFFTImage = refVolume.AsFFT(true);
                        volumeWarpRFFTImage.AsAmplitudes().WriteMRC(@$"{directory}\volumeWarpRFFTImage.mrc", true);
                        TorchTensor foo = new_volume.rfftn(new long[] { 2, 3, 4 });
                        TorchTensor foo_shifted = foo.fftshift(new long[] { 2, 3 });
                        Image fft_abs = new Image(refVolume.Dims, true);
                        GPU.CopyDeviceToDevice(foo.Abs().DataPtr(), fft_abs.GetDevice(Intent.Write), fft_abs.ElementsReal);
                        fft_abs.WriteMRC(@$"{directory}\volumeRFFTImage.mrc", true);

                        GPU.CopyDeviceToDevice(foo_shifted.Abs().DataPtr(), fft_abs.GetDevice(Intent.Write), fft_abs.ElementsReal);
                        fft_abs.WriteMRC(@$"{directory}\volumeRFFTShiftedImage.mrc", true);
                    }*/

                    {
                        Image imageRfftVolume = new Image(new int3(boxLength, boxLength, boxLength), true);
                        TorchTensor rfftn = volume.rfftn(new long[] { 2, 3, 4 });
                        GPU.CopyDeviceToDevice(rfftn.Abs().DataPtr(), imageRfftVolume.GetDevice(Intent.Write), imageRfftVolume.ElementsReal);
                        imageRfftVolume.WriteMRC(@$"{directory}\imageRfftVolumeAbs.mrc", true);

                        Image Re = new Image(new int3(boxLength, boxLength, boxLength), true);

                        GPU.CopyDeviceToDevice(rfftn.Real().Clone().DataPtr(), Re.GetDevice(Intent.Write), Re.ElementsReal);


                        Image Imag = new Image(new int3(boxLength, boxLength, boxLength), true);

                        GPU.CopyDeviceToDevice(rfftn.Imag().Clone().DataPtr(), Imag.GetDevice(Intent.Write), Imag.ElementsReal);

                        Image.Stack(new Image[] { Re, Imag }).WriteMRC(@$"{directory}\imageRfftVolumeStacked.mrc", true);

                    }
                    {
                        Image imagefftVolume = new Image(new int3(boxLength, boxLength, boxLength), true);
                        GPU.CopyDeviceToDevice(fftVolume.Abs().DataPtr(), imagefftVolume.GetDevice(Intent.Write), imagefftVolume.ElementsReal);
                        imagefftVolume.WriteMRC(@$"{directory}\imagefftVolumeAbs.mrc", true);

                        Image imagefftVolumeReal = new Image(new int3(boxLength, boxLength, boxLength));
                        GPU.CopyDeviceToDevice(fftVolume.Real().Clone().DataPtr(), imagefftVolumeReal.GetDevice(Intent.Write), imagefftVolumeReal.ElementsReal);

                        Image imagefftVolumeImag = new Image(new int3(boxLength, boxLength, boxLength));
                        GPU.CopyDeviceToDevice(fftVolume.Imag().Clone().DataPtr(), imagefftVolumeImag.GetDevice(Intent.Write), imagefftVolumeImag.ElementsReal);
  
                        Image.Stack(new Image[] { imagefftVolumeReal, imagefftVolumeImag }).WriteMRC(@$"{directory}\imagefftVolumeStacked.mrc", true);

                    }

                    {
                        Image imagefftShiftedVolume = new Image(new int3(boxLength, boxLength, boxLength), true);
                        GPU.CopyDeviceToDevice(fftShiftedVolume.Abs().DataPtr(), imagefftShiftedVolume.GetDevice(Intent.Write), imagefftShiftedVolume.ElementsReal);
                        imagefftShiftedVolume.WriteMRC(@$"{directory}\imagefftShiftedVolumeAbs.mrc", true);

                        Image imagefftShiftedVolumeReal = new Image(new int3(boxLength, boxLength, boxLength), true);
                        GPU.CopyDeviceToDevice(fftShiftedVolume.Real().Clone().DataPtr(), imagefftShiftedVolumeReal.GetDevice(Intent.Write), imagefftShiftedVolumeReal.ElementsReal);

                        Image imagefftShiftedVolumeImag = new Image(new int3(boxLength, boxLength, boxLength), true);
                        GPU.CopyDeviceToDevice(fftShiftedVolume.Imag().Clone().DataPtr(), imagefftShiftedVolumeImag.GetDevice(Intent.Write), imagefftShiftedVolumeImag.ElementsReal);
                        Image.Stack(new Image[] { imagefftShiftedVolumeReal, imagefftShiftedVolumeImag }).WriteMRC(@$"{directory}\imagefftShiftedVolumeStacked.mrc", true);
                    }

                    {
                        Image imagefftSlice = new Image(new int3(boxLength, boxLength, 1), true);
                        GPU.CopyDeviceToDevice(fftSlice.Abs().DataPtr(), imagefftSlice.GetDevice(Intent.Write), imagefftSlice.ElementsReal); ;
                        imagefftSlice.WriteMRC(@$"{directory}\imagefftSliceAbs.mrc", true);

                        Image imagefftSliceReal = new Image(new int3(boxLength, boxLength, 1), true);
                        GPU.CopyDeviceToDevice(fftSlice.Real().Clone().DataPtr(), imagefftSliceReal.GetDevice(Intent.Write), imagefftSliceReal.ElementsReal); ;
                    
                        Image imagefftSliceImag = new Image(new int3(boxLength, boxLength, 1), true);
                        GPU.CopyDeviceToDevice(fftSlice.Imag().Clone().DataPtr(), imagefftSliceImag.GetDevice(Intent.Write), imagefftSliceImag.ElementsReal); ;
                        Image.Stack(new Image[] { imagefftSliceReal, imagefftSliceImag }).WriteMRC(@$"{directory}\imagefftSliceStacked.mrc", true);
                    }
                    {
                        Image imageSlice = new Image(new int3(boxLength, boxLength, 1));
                        GPU.CopyDeviceToDevice(Slice.DataPtr(), imageSlice.GetDevice(Intent.Write), imageSlice.ElementsReal);
                        imageSlice.RemapFromFT();
                        imageSlice.WriteMRC(@$"{directory}\imageSlice.mrc", true);
                    }
                }

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
