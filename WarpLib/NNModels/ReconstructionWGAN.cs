using System;

using System.IO;
using System.Linq;

using Warp.Tools;
using TorchSharp.Tensor;
using TorchSharp.NN;
using static TorchSharp.NN.Modules;
using TorchSharp;

namespace Warp.NNModels
{
    public class ReconstructionWGAN
    {
        public readonly int2 BoxDimensions;
        public readonly int BatchSize = 8;
        public readonly int DeviceBatch = 8;
        public readonly int[] Devices;
        public readonly int NDevices;

        private ReconstructionWGANGenerator[] Generators;
        private ReconstructionWGANDiscriminator[] Discriminators;

        private TorchTensor[] TensorTrueImages;
        private TorchTensor[] TensorFakeImages;
        private TorchTensor[] TensorCTF;
        private TorchTensor[] TensorNoise;
        private TorchTensor[] TensorAngles;

        private TorchTensor[] TensorOne;
        private TorchTensor[] TensorMinusOne;

        private TorchTensor[] TensorMask;
        private TorchTensor[] TensorBinaryMask;
        private TorchTensor[] TensorMaxMask;
        private TorchTensor[] TensorVolumeMask;
        private TorchTensor[] TensorGaussian;

        private Optimizer OptimizerGen;
        private Optimizer OptimizerGenVolume;
        private Optimizer OptimizerDisc;

        private Image ResultPredictedGen;
        private Image ResultPredictedDisc;
        private Image ResultPredictedNoisy;
        private float[] ResultLoss = new float[1];
        private float[] ResultLossDiscReal = new float[1];
        private float[] ResultLossDiscFake = new float[1];

        private double sigmaShift = 0;

        private bool IsDisposed = false;
        private bool doShift = false;

        private double GenVolumeBoost = 1;
        private double GenBoost = 1;
        private double generator_grad_clip_val = 1e4f;
        private double discriminator_grad_clip_val = 1e8f;
        private double lambdaOutsideMask = 10;

        private int3 OversampledBoxDimensions;
        private bool doNormalizeInput = true;
        private bool doMaskProjections = false;

        public double SigmaShift { get => sigmaShift; set => sigmaShift = value; }
        public int numGenIt;
        public ReconstructionWGAN(int2 boxDimensions, int[] devices, int batchSize = 8, Image discriminatorImage=null, Image initialVolume = null)
        {
            Devices = devices;
            NDevices = Devices.Length;

            BatchSize = Math.Max(batchSize, NDevices);
            DeviceBatch = BatchSize / NDevices;
            if (BatchSize % NDevices != 0)
                throw new Exception("Batch size must be divisible by the number of devices.");

            BoxDimensions = boxDimensions;
            OversampledBoxDimensions = new(boxDimensions);// boxDimensions*4;//boxDimensions*2;

            Generators = new ReconstructionWGANGenerator[NDevices];
            Discriminators = new ReconstructionWGANDiscriminator[NDevices];

            TensorTrueImages = new TorchTensor[NDevices];
            TensorGaussian = new TorchTensor[NDevices];
            TensorFakeImages = new TorchTensor[NDevices];
            TensorCTF = new TorchTensor[NDevices];
            TensorNoise = new TorchTensor[NDevices];
            TensorAngles = new TorchTensor[NDevices];

            TensorOne = new TorchTensor[NDevices];
            TensorMinusOne = new TorchTensor[NDevices];

            TensorMask = new TorchTensor[NDevices];
            TensorBinaryMask = new TorchTensor[NDevices];
            TensorMaxMask = new TorchTensor[NDevices];
            TensorVolumeMask = new TorchTensor[NDevices];
            numGenIt = 0;

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                int DeviceID = Devices[i];

                TorchTensor discrimIm = Float32Tensor.Empty(new long[] { 1, discriminatorImage.Dims.Z, discriminatorImage.Dims.Y, discriminatorImage.Dims.X }, DeviceType.CUDA, DeviceID);
                GPU.CopyDeviceToDevice(discriminatorImage.GetDevice(Intent.Read), discrimIm.DataPtr(), discriminatorImage.ElementsReal);
                Discriminators[i] = ReconstructionWGANDiscriminator(boxDimensions.X, doNormalizeInput);
                Discriminators[i].ToCuda(DeviceID);

                TensorTrueImages[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorGaussian[i] = Float32Tensor.Zeros(new long[] { 1, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorFakeImages[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorNoise[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorCTF[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X / 2 + 1 }, DeviceType.CUDA, DeviceID);
                TensorAngles[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 3 }, DeviceType.CUDA, DeviceID);

                TensorOne[i] = Float32Tensor.Ones(new long[] { }, DeviceType.CUDA, DeviceID);
                TensorMinusOne[i] = Float32Tensor.Ones(new long[] { }, DeviceType.CUDA, DeviceID);
                {
                    GPU.CopyHostToDevice(new float[] { -1f }, TensorMinusOne[i].DataPtr(), 1);
                }

                TensorMask[i] = Float32Tensor.Ones(new long[] { 1, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                {
                    Image Mask = new Image(new int3(BoxDimensions.X, BoxDimensions.Y, 1));
                    Mask.Fill(1);
                    Mask.MaskSpherically(BoxDimensions.X / 2+2* BoxDimensions.X / 8, 0, false);

                    GPU.CopyDeviceToDevice(Mask.GetDevice(Intent.Read), TensorMask[i].DataPtr(), Mask.ElementsReal);

                    Mask.Dispose();
                }
                TensorBinaryMask[i] = Float32Tensor.Ones(new long[] { 1, OversampledBoxDimensions.X, OversampledBoxDimensions.Y, OversampledBoxDimensions.X }, DeviceType.CUDA, DeviceID);
                {
                    Image Mask = new Image(new int3(OversampledBoxDimensions.X, OversampledBoxDimensions.Y, OversampledBoxDimensions.Y));
                    double rLim = OversampledBoxDimensions.X/2;// 0.5 *(OversampledBoxDimensions.X / 2.0d + 2.0d * OversampledBoxDimensions.X / 8.0d);
                    Mask.TransformValues((x, y, z, val) =>
                    {
                        double xx = Math.Pow(x -OversampledBoxDimensions.X / 2.0d, 2);
                        double yy = Math.Pow(y -OversampledBoxDimensions.X / 2.0d, 2);
                        double zz = Math.Pow(z- OversampledBoxDimensions.X / 2.0d, 2);
                        double r = Math.Sqrt(xx + yy + zz);

                        return (float)(r>rLim?0:1);
                    });
                    //Mask.WriteMRC("BinaryMask.mrc", true);
                    GPU.CopyDeviceToDevice(Mask.GetDevice(Intent.Read), TensorBinaryMask[i].DataPtr(), Mask.ElementsReal);

                    Mask.Dispose();
                }

                TensorMaxMask[i] = Float32Tensor.Ones(new long[] { 1, OversampledBoxDimensions.X, OversampledBoxDimensions.Y, OversampledBoxDimensions.X }, DeviceType.CUDA, DeviceID);
                {
                    Image Mask = new Image(new int3(OversampledBoxDimensions.X, OversampledBoxDimensions.Y, OversampledBoxDimensions.Y));
                    double rLim = OversampledBoxDimensions.X/2;// 0.5 *(OversampledBoxDimensions.X / 2.0d + 2.0d * OversampledBoxDimensions.X / 8.0d);
                    Mask.TransformValues((x, y, z, val) => 
                    {
                        double xx = Math.Pow(x - OversampledBoxDimensions.X / 2.0d, 2);
                        double yy = Math.Pow(y- OversampledBoxDimensions.X / 2.0d, 2);
                        double zz = Math.Pow(z -OversampledBoxDimensions.X / 2.0d, 2);
                        double r = Math.Sqrt(xx + yy + zz);

                        return (float)(r<rLim?1 - r / rLim:0); 
                    });
                    //Mask.WriteMRC("TensorMaxMask.mrc", true);
                    GPU.CopyDeviceToDevice(Mask.GetDevice(Intent.Read), TensorMaxMask[i].DataPtr(), Mask.ElementsReal);

                    Mask.Dispose();
                }

                TensorVolumeMask[i] = Float32Tensor.Ones(new long[] { 1, OversampledBoxDimensions.X, OversampledBoxDimensions.Y, OversampledBoxDimensions.X }, DeviceType.CUDA, DeviceID);
                {
                    Image Mask = new Image(new int3(OversampledBoxDimensions.X, OversampledBoxDimensions.Y, OversampledBoxDimensions.X));
                    Mask.Fill(1);
                    Mask.MaskSpherically(OversampledBoxDimensions.X / 2+ 2* OversampledBoxDimensions.X / 8, 0, true);
                    //Mask.WriteMRC(@"D:\GAN_recon_polcompl\mask.mrc", true);
                    GPU.CopyDeviceToDevice(Mask.GetDevice(Intent.Read), TensorVolumeMask[i].DataPtr(), Mask.ElementsReal);

                    Mask.Dispose();
                }

                TorchTensor volume = Float32Tensor.Zeros(new long[] { 1, 1, boxDimensions.X, boxDimensions.X, boxDimensions.X }, DeviceType.CUDA, DeviceID);
                if (initialVolume != null)
                {
                    volume = TensorExtensionMethods.ToTorchTensor(initialVolume.GetHostContinuousCopy(), new long[] { 1, 1, boxDimensions.X, boxDimensions.X, boxDimensions.X }).ToDevice(DeviceType.CUDA, DeviceID);
                }
                Generators[i] = ReconstructionWGANGenerator(volume, BoxDimensions.X);
                Generators[i].ToCuda(DeviceID);
            }, null);


            OptimizerGenVolume = Optimizer.Adam(Generators[0].GetParameters().Take(1), 0.01, 1e-8);
            OptimizerGenVolume.SetBetasAdam(0.5, 0.9);
            OptimizerGen = Optimizer.Adam(Generators[0].GetParameters().Skip(1), 0.01, 1e-8);
            OptimizerGen.SetBetasAdam(0.5, 0.9);
            OptimizerDisc = Optimizer.Adam(Discriminators[0].GetParameters(), 0.01, 1e-8);
            OptimizerDisc.SetBetasAdam(0.5, 0.9);

            ResultPredictedGen = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, BatchSize));
            ResultPredictedDisc = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, BatchSize));
            ResultPredictedNoisy = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, BatchSize));
        }

        private void ScatterData(Image src, TorchTensor[] dest)
        {
            src.GetDevice(Intent.Read);

            for (int i = 0; i < NDevices; i++)
                GPU.CopyDeviceToDevice(src.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       dest[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.Elements());
        }

        private void SyncParams()
        {
            for (int i = 1; i < NDevices; i++)
            {
                Generators[0].SynchronizeTo(Generators[i], Devices[i]);
                Discriminators[0].SynchronizeTo(Discriminators[i], Devices[i]);
            }
        }

        private void GatherGrads()
        {
            for (int i = 1; i < NDevices; i++)
            {
                Generators[0].GatherGrad(Generators[i]);
                Discriminators[0].GatherGrad(Discriminators[i]);
            }
        }

        public Image getVolume()
        {
            var tensorVolume = Generators[0].Get_Volume();
            var imageVolume = new Image(new int3(this.BoxDimensions.X));
            GPU.CopyDeviceToDevice(tensorVolume.DataPtr(), imageVolume.GetDevice(Intent.Write), imageVolume.ElementsReal);
            imageVolume.FreeDevice();
            return imageVolume;
        }

        public float getSigma()
        {

            var tensorShift = Generators[0].GetParameters()[1];

            float[] shiftSotre = new float[1];
            GPU.CopyDeviceToHost(tensorShift.DataPtr(), shiftSotre, 1);

            return shiftSotre[0];
        }

        public void set_discriminator_grad_clip_val(double clip_val)
        {
            discriminator_grad_clip_val = clip_val;
        }

        public void set_generator_grad_clip_val(double clip_val)
        {
            generator_grad_clip_val = clip_val;
        }
        public void TrainGeneratorParticle(float[] angles,
                                           Image imagesCTF,
                                           Image imagesReal,
                                           Image imagesnoise,
                                           Image GaussianKernel,
                                           float learningRate,
                                           out Image prediction,
                                           out Image predictionNoisy,
                                           out float[] loss,
                                           out double gradNorm)
        {
            OptimizerGen.SetLearningRateAdam(learningRate * GenBoost);
            OptimizerGenVolume.SetLearningRateAdam(learningRate * GenVolumeBoost);
            OptimizerGen.ZeroGrad();
            /*if (GaussianKernel != null)
                GaussianKernel.WriteMRC("GaussianinGenerator.mrc", true);*/
            SyncParams();
            ResultPredictedGen.GetDevice(Intent.Write);
            ResultPredictedNoisy.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                Generators[i].Train();
                Generators[i].ZeroGrad();
                GPU.CopyDeviceToDevice(imagesReal.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorTrueImages[i].DataPtr(),
                                       BoxDimensions.Elements() * DeviceBatch);
                GPU.CopyDeviceToDevice(imagesnoise.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorNoise[i].DataPtr(),
                                       BoxDimensions.Elements() * DeviceBatch);
                GPU.CopyDeviceToDevice(imagesCTF.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorCTF[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.ElementsFFT());
                if(GaussianKernel != null)
                    GPU.CopyDeviceToDevice(GaussianKernel.GetDevice(Intent.Read), TensorGaussian[i].DataPtr(), GaussianKernel.ElementsReal);

                GPU.CopyHostToDevice(angles, TensorAngles[i].DataPtr(), DeviceBatch * 3);

                using (TorchTensor Prediction = Generators[i].Forward(TensorAngles[i], doShift))
                using (TorchTensor PredictionFT = Prediction.rfftn(new long[] { 2, 3 }))
                using (TorchTensor PredictionFTConv = PredictionFT.Mul(TensorCTF[i]))
                using (TorchTensor PredictionConv = PredictionFTConv.irfftn(new long[] { 2, 3 }))
                //using (TorchTensor PredictionNoisy = Generators[i].ApplyNoise(PredictionConv, TensorCTF[i]))
                using (TorchTensor PredictionNoisy = PredictionConv.Add(TensorNoise[i]))
                //using (TorchTensor PredictionNoisyMean = doNormalizeInput ? PredictionNoisy.Mean(new long[] { 2, 3 }, true) : null)
                //using (TorchTensor PredictionNoisyStd = doNormalizeInput ? PredictionNoisy.Std(new long[] { 2, 3 }, true, true) : null)
                //using (TorchTensor PredictionNoisyNormalized = doNormalizeInput ? (PredictionNoisy - PredictionNoisyMean) / (PredictionNoisyStd + 1e-4) : PredictionNoisy)
                using (TorchTensor PredictionNoisyMasked = doMaskProjections ? PredictionNoisy.Mul(TensorMask[i]) : PredictionNoisy)
                using (TorchTensor fft = GaussianKernel != null ? PredictionNoisyMasked.rfftn(new long[] { 2, 3 }) : null)
                using (TorchTensor conv = GaussianKernel != null ? fft.Mul(TensorGaussian[i].rfftn(new long[] { 2, 3 })) : null)
                using (TorchTensor ifft = GaussianKernel != null ? conv.irfftn(new long[] { 2, 3 }) : null)
                using (TorchTensor IsItReal = Discriminators[i].Forward(GaussianKernel != null ? ifft : PredictionNoisyMasked))
                using (TorchTensor Loss = ((-1) * IsItReal).Mean())
                //using (TorchTensor Loss = (PredictionNoisyMasked - TensorTrueImages[i]).Pow(2).Mean())
                {

                    GPU.CopyDeviceToDevice(Prediction.DataPtr(),
                                           ResultPredictedGen.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                           DeviceBatch * (int)BoxDimensions.Elements());


                    GPU.CopyDeviceToDevice(PredictionNoisyMasked.DataPtr(),
                                           ResultPredictedNoisy.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                           DeviceBatch * (int)BoxDimensions.Elements());
                    //imagesCTF.WriteMRC("imagesCTF.mrc", true);
                    //imagesReal.WriteMRC("imagesReal.mrc", true);
                    //imagesnoise.WriteMRC("imagesnoise.mrc", true);
                    //ResultPredictedGen.WriteMRC("Prediction.mrc", true);
                    //ResultPredictedNoisy.WriteMRC("PredictionNoisyMasked.mrc", true);

                    if (i == 0)
                        GPU.CopyDeviceToHost(Loss.DataPtr(), ResultLoss, 1);

                    Loss.Backward();
                }
                /*
                using (TorchTensor thisVolume = Generators[i].GetParameters()[0])
                using (TorchTensor maskInv = (-1) * TensorVolumeMask[i] + 1)
                using (TorchTensor outsideMask = thisVolume * maskInv)
                using (TorchTensor outsideMaskWeighted = outsideMask * lambdaOutsideMask)
                using (TorchTensor penaltyAbs = outsideMaskWeighted.Pow(2))
                using (TorchTensor penalty = penaltyAbs.Sum())
                {
                    penalty.Backward();
                }*/
            }, null);

            GatherGrads();
            gradNorm = Generators[0].Clip_Gradients(generator_grad_clip_val);
            gradNorm = 0.0d;
            OptimizerGen.Step();
            OptimizerGenVolume.Step();
            prediction = ResultPredictedGen;
            //prediction.WriteMRC("PredictionInTrainingFunc.mrc", true);
            prediction.FreeDevice();
            predictionNoisy = ResultPredictedNoisy;
            //predictionNoisy.WriteMRC("predictionNoisyInTrainingFunc.mrc", true);
            predictionNoisy.FreeDevice();
            loss = ResultLoss;
            double multiplicator = Math.Min((double)numGenIt / 1000, 1.0) * (-2.0 / 100);
            Generators[0].ApplY_Volume_Mask(TensorBinaryMask[0], TensorMaxMask[0], multiplicator); 

            numGenIt++;
        }

        public void TrainDiscriminatorParticle(float[] angles,
                                               Image imagesReal,
                                               Image imagesCTF,
                                               Image imagesNoise,
                                               Image GaussianKernel,
                                               float learningRate,
                                               float penaltyLambda,
                                               out Image prediction,
                                               out float[] lossWasserstein,
                                               out float[] lossReal,
                                               out float[] lossFake,
                                               out double gradNorm)
        {
            OptimizerDisc.SetLearningRateAdam(learningRate);
            OptimizerDisc.ZeroGrad();
            /*if(GaussianKernel != null)
            GaussianKernel.WriteMRC("GaussianinDiscriminator.mrc", true);*/
            SyncParams();
            ResultPredictedDisc.GetDevice(Intent.Write);
            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                Discriminators[i].Train();
                Discriminators[i].ZeroGrad();

                GPU.CopyDeviceToDevice(imagesReal.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorTrueImages[i].DataPtr(),
                                       BoxDimensions.Elements() * DeviceBatch);
                GPU.CopyDeviceToDevice(imagesNoise.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorNoise[i].DataPtr(),
                                       BoxDimensions.Elements() * DeviceBatch);
                GPU.CopyDeviceToDevice(imagesCTF.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorCTF[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.ElementsFFT());

                float LossWasserstein = 0;

                //using (TorchTensor TrueImageMean = doNormalizeInput ? TensorTrueImages[i].Mean(new long[] { 2, 3 }, true) : null)
                //using (TorchTensor TrueImageStd = doNormalizeInput ? TensorTrueImages[i].Std(new long[] { 2, 3 }, true, true) : null)
                //using (TorchTensor TrueImageNormalized = doNormalizeInput ? ((TensorTrueImages[i] - TrueImageMean) / (TrueImageStd + 1e-4)) : null)
                using (TorchTensor realInput = doMaskProjections ? TensorTrueImages[i].Mul(TensorMask[i]) : null)
                using (TorchTensor fft = GaussianKernel != null ? (doMaskProjections ? realInput : TensorTrueImages[i]).rfftn(new long[] { 2, 3 }) : null)
                using (TorchTensor conv = GaussianKernel != null ? fft.Mul(TensorGaussian[i].rfftn(new long[] { 2, 3 })) : null)
                using (TorchTensor ifft = GaussianKernel != null ? conv.irfftn(new long[] { 2, 3 }) : null)
                using (TorchTensor IsTrueReal = Discriminators[i].Forward(GaussianKernel != null ? ifft : (doMaskProjections ? realInput : TensorTrueImages[i])))
                using (TorchTensor IsTrueRealNeg = IsTrueReal * (-1))
                using (TorchTensor LossReal = IsTrueRealNeg.Mean())
                {
                    if (i == 0)
                    {
                        GPU.CopyDeviceToHost(LossReal.DataPtr(), ResultLossDiscReal, 1);
                        LossWasserstein = ResultLossDiscReal[0];
                    }

                    GPU.CopyHostToDevice(angles, TensorAngles[i].DataPtr(), DeviceBatch * 3);
                    using (TorchTensor Prediction = Generators[i].Forward(TensorAngles[i], doShift))
                    using (TorchTensor PredictionFT = Prediction.rfftn(new long[] { 2, 3 }))
                    using (TorchTensor PredictionFTConv = PredictionFT.Mul(TensorCTF[i]))
                    using (TorchTensor PredictionConv = PredictionFTConv.irfftn(new long[] { 2, 3 }))
                    //using (TorchTensor PredictionNoisy = Generators[i].ApplyNoise(PredictionConv, TensorCTF[i]))
                    using (TorchTensor PredictionNoisy = PredictionConv.Add(TensorNoise[i]))
                    //using (TorchTensor PredictionNoisyMean = doNormalizeInput ? PredictionNoisy.Mean(new long[] { 2, 3 }, true) : null)
                    //using (TorchTensor PredictionNoisyStd = doNormalizeInput ? PredictionNoisy.Std(new long[] { 2, 3 }, true, true) : null)
                    //using (TorchTensor PredictionNoisyNormalized = doNormalizeInput ? (PredictionNoisy - PredictionNoisyMean) / (PredictionNoisyStd + 1e-4) : null)
                    using (TorchTensor PredictedInput = doMaskProjections ?
                          ( PredictionConv.Mul(TensorMask[i]))
                        : (PredictionConv))
                    using (TorchTensor PredictedInputDetached = PredictedInput.Detach())
                    using (TorchTensor fftT = GaussianKernel != null ? PredictedInputDetached.rfftn(new long[] { 2, 3 }) : null)
                    using (TorchTensor convT = GaussianKernel != null ? fftT.Mul(TensorGaussian[i].rfftn(new long[] { 2, 3 })) : null)
                    using (TorchTensor ifftT = GaussianKernel != null ? convT.irfftn(new long[] { 2, 3 }) : null)
                    using (TorchTensor IsFakeReal = Discriminators[i].Forward(GaussianKernel != null ? ifftT : PredictedInputDetached))
                    using (TorchTensor LossFake = IsFakeReal.Mean())
                    {
                        GPU.CopyDeviceToDevice(PredictedInput.DataPtr(),
                                               ResultPredictedDisc.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                               DeviceBatch * (int)BoxDimensions.Elements());
                        if (i == 0)
                        {
                            GPU.CopyDeviceToHost(LossFake.DataPtr(), ResultLossDiscFake, 1);
                            LossWasserstein = LossWasserstein + ResultLossDiscFake[0];
                            ResultLoss[0] = LossWasserstein;
                        }
                        LossReal.Backward();
                        LossFake.Backward();
                        if (penaltyLambda > 0.0)
                        {
                            using (TorchTensor Penalty = Discriminators[i].PenalizeGradient(doMaskProjections ? realInput : TensorTrueImages[i], PredictedInput, penaltyLambda))
                            {
                                Penalty.Backward();
                            }
                        }
                    }
                }
            }, null);

            GatherGrads();
            gradNorm = Discriminators[0].Clip_Gradients(discriminator_grad_clip_val);
            OptimizerDisc.Step();

            prediction = ResultPredictedDisc;
            prediction.FreeDevice();
            lossWasserstein = ResultLoss;
            lossReal = ResultLossDiscReal;
            lossFake = ResultLossDiscFake;
        }

        public void Save(string path)
        {
            Directory.CreateDirectory(Helper.PathToFolder(path));

            Generators[0].Save(path + ".gen");
            Discriminators[0].Save(path + ".disc");
        }

        public void Load(string path)
        {
            for (int i = 0; i < NDevices; i++)
            {
                Generators[i].Load(path + ".gen", DeviceType.CUDA, Devices[i]);
                Discriminators[i].Load(path + ".disc", DeviceType.CUDA, Devices[i]);
            }
        }

        ~ReconstructionWGAN()
        {
            Dispose();
        }

        public void Dispose()
        {
            lock (this)
            {
                if (!IsDisposed)
                {
                    IsDisposed = true;

                    ResultPredictedGen.Dispose();

                    for (int i = 0; i < NDevices; i++)
                    {
                        TensorTrueImages[i].Dispose();
                        TensorCTF[i].Dispose();

                        TensorOne[i].Dispose();
                        TensorMinusOne[i].Dispose();

                        Generators[i].Dispose();
                        Discriminators[i].Dispose();
                    }

                    OptimizerGen.Dispose();
                    OptimizerDisc.Dispose();
                }
            }
        }
    }
}
