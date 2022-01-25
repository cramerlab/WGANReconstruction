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
        private TorchTensor[] TensorAngles;

        private TorchTensor[] TensorOne;
        private TorchTensor[] TensorMinusOne;

        private TorchTensor[] TensorMask;
        private TorchTensor[] TensorVolumeMask;

        private Optimizer OptimizerGen;
        private Optimizer OptimizerGenVolume;
        private Optimizer OptimizerDisc;

        private Image ResultPredicted;
        private Image ResultPredictedNoisy;
        private float[] ResultLoss = new float[1];
        private float[] ResultLossDiscReal = new float[1];
        private float[] ResultLossDiscFake = new float[1];

        private double sigmaShift = 0;

        private bool IsDisposed = false;
        private bool doShift = false;

        private double GenVolumeBoost = 10;
        private double GenBoost = 1;
        private double generator_grad_clip_val = 1e4f;
        private double discriminator_grad_clip_val = 1e8f;
        private double lambdaOutsideMask = 10;

        private int3 OversampledBoxDimensions;
        private bool doNormalizeInput = true;
        private bool doMaskProjections = false;

        public double SigmaShift { get => sigmaShift; set => sigmaShift = value; }

        public ReconstructionWGAN(int2 boxDimensions, int[] devices, int batchSize = 8, Image initialVolume = null)
        {
            Devices = devices;
            NDevices = Devices.Length;

            BatchSize = Math.Max(batchSize, NDevices);
            DeviceBatch = BatchSize / NDevices;
            if (BatchSize % NDevices != 0)
                throw new Exception("Batch size must be divisible by the number of devices.");

            BoxDimensions = boxDimensions;
            OversampledBoxDimensions = new(128);// boxDimensions*4;//boxDimensions*2;

            Generators = new ReconstructionWGANGenerator[NDevices];
            Discriminators = new ReconstructionWGANDiscriminator[NDevices];

            TensorTrueImages = new TorchTensor[NDevices];
            TensorFakeImages = new TorchTensor[NDevices];
            TensorCTF = new TorchTensor[NDevices];
            TensorAngles = new TorchTensor[NDevices];

            TensorOne = new TorchTensor[NDevices];
            TensorMinusOne = new TorchTensor[NDevices];

            TensorMask = new TorchTensor[NDevices];
            TensorVolumeMask = new TorchTensor[NDevices];

            

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                int DeviceID = Devices[i];


                Discriminators[i] = ReconstructionWGANDiscriminator(boxDimensions.X);
                Discriminators[i].ToCuda(DeviceID);

                TensorTrueImages[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorFakeImages[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
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
                TensorVolumeMask[i] = Float32Tensor.Ones(new long[] { 1, OversampledBoxDimensions.X, OversampledBoxDimensions.Y, OversampledBoxDimensions.X }, DeviceType.CUDA, DeviceID);
                {
                    Image Mask = new Image(new int3(OversampledBoxDimensions.X, OversampledBoxDimensions.Y, OversampledBoxDimensions.X));
                    Mask.Fill(1);
                    Mask.MaskSpherically(OversampledBoxDimensions.X / 2+ 2* OversampledBoxDimensions.X / 8, 0, true);
                    Mask.WriteMRC(@"D:\GAN_recon_polcompl\mask.mrc", true);
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

            ResultPredicted = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, BatchSize));
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
                                           float learningRate,
                                           out Image prediction,
                                           out Image predictionNoisy,
                                           out float[] loss,
                                           out double gradNorm)
        {
            OptimizerGen.SetLearningRateAdam(learningRate * GenBoost);
            OptimizerGenVolume.SetLearningRateAdam(learningRate * GenVolumeBoost);
            OptimizerGen.ZeroGrad();

            SyncParams();
            ResultPredicted.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                Generators[i].Train();
                Generators[i].ZeroGrad();
                GPU.CopyDeviceToDevice(imagesReal.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorTrueImages[i].DataPtr(),
                                       BoxDimensions.Elements() * DeviceBatch);
                GPU.CopyDeviceToDevice(imagesCTF.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorCTF[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.ElementsFFT());

                GPU.CopyHostToDevice(angles, TensorAngles[i].DataPtr(), DeviceBatch * 3);

                using (TorchTensor Prediction = Generators[i].Forward(TensorAngles[i], doShift))
                using (TorchTensor PredictionFT = Prediction.rfftn(new long[] { 2, 3 }))
                using (TorchTensor PredictionFTConv = PredictionFT.Mul(TensorCTF[i]))
                using (TorchTensor PredictionConv = PredictionFTConv.irfftn(new long[] { 2, 3 }))
                using (TorchTensor PredictionNoisy = Generators[i].ApplyNoise(PredictionConv, TensorCTF[i]))
                using (TorchTensor PredictionNoisyMean = doNormalizeInput?PredictionNoisy.Mean(new long[] { 2, 3 }, true):null) 
                using (TorchTensor PredictionNoisyStd = doNormalizeInput?PredictionNoisy.Std(new long[] { 2, 3 }, true, true):null)
                using (TorchTensor PredictionNoisyNormalized = doNormalizeInput?(PredictionNoisy - PredictionNoisyMean) /(PredictionNoisyStd + 1e-4): PredictionNoisy)
                using (TorchTensor PredictionNoisyMasked = doMaskProjections? PredictionNoisyNormalized.Mul(TensorMask[i]):PredictionNoisyNormalized)
                using (TorchTensor IsItReal = Discriminators[i].Forward(PredictionNoisyMasked))
                using (TorchTensor Loss = ((-1) * IsItReal).Mean())
                {

                    GPU.CopyDeviceToDevice(Prediction.DataPtr(),
                                           ResultPredicted.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                           DeviceBatch * (int)BoxDimensions.Elements());
                    GPU.CopyDeviceToDevice(PredictionNoisyMasked.DataPtr(),
                                           ResultPredictedNoisy.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                           DeviceBatch * (int)BoxDimensions.Elements());
                    if (i == 0)
                        GPU.CopyDeviceToHost(Loss.DataPtr(), ResultLoss, 1);

                    Loss.Backward(TensorOne[i]);
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
            OptimizerGen.Step();
            OptimizerGenVolume.Step();
            prediction = ResultPredicted;
            prediction.FreeDevice();
            predictionNoisy = ResultPredictedNoisy;
            predictionNoisy.FreeDevice();
            loss = ResultLoss;
        }

        public void TrainDiscriminatorParticle(float[] angles,
                                               Image imagesReal,
                                               Image imagesCTF,
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

            SyncParams();
            ResultPredicted.GetDevice(Intent.Write);
            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                Discriminators[i].Train();
                Discriminators[i].ZeroGrad();

                GPU.CopyDeviceToDevice(imagesReal.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorTrueImages[i].DataPtr(),
                                       BoxDimensions.Elements() * DeviceBatch);
                GPU.CopyDeviceToDevice(imagesCTF.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorCTF[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.ElementsFFT());

                float LossWasserstein = 0;

                using (TorchTensor TrueImageMean = doNormalizeInput ? TensorTrueImages[i].Mean(new long[] { 2, 3 }, true) : null)
                using (TorchTensor TrueImageStd = doNormalizeInput ? TensorTrueImages[i].Std(new long[] { 2, 3 }, true, true) : null)
                using (TorchTensor TrueImageNormalized = doNormalizeInput ? ((TensorTrueImages[i] - TrueImageMean) / (TrueImageStd + 1e-4)) : null)
                using (TorchTensor realInput = doMaskProjections ? 
                    (doNormalizeInput ? TrueImageNormalized.Mul(TensorMask[i]) : TensorTrueImages[i].Mul(TensorMask[i]))
                  : (doNormalizeInput ? TrueImageNormalized : TensorTrueImages[i]))
                using (TorchTensor IsTrueReal = Discriminators[i].Forward(realInput))
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
                    using (TorchTensor PredictionNoisy = Generators[i].ApplyNoise(PredictionConv, TensorCTF[i]))
                    using (TorchTensor PredictionNoisyMean = doNormalizeInput ? PredictionNoisy.Mean(new long[] { 2, 3 }, true) : null)
                    using (TorchTensor PredictionNoisyStd = doNormalizeInput ? PredictionNoisy.Std(new long[] { 2, 3 }, true, true) : null)
                    using (TorchTensor PredictionNoisyNormalized = doNormalizeInput ? (PredictionNoisy - PredictionNoisyMean) / (PredictionNoisyStd + 1e-4) : null)
                    using (TorchTensor PredictedInput = doMaskProjections ?
                          (doNormalizeInput ? PredictionNoisyNormalized.Mul(TensorMask[i]) : PredictionConv.Mul(TensorMask[i]))
                        : (doNormalizeInput ? PredictionNoisyNormalized : PredictionConv))
                    using (TorchTensor PredictedInputDetached = PredictedInput.Detach())
                    using (TorchTensor IsFakeReal = Discriminators[i].Forward(PredictedInputDetached))
                    using (TorchTensor LossFake = IsFakeReal.Mean())
                    {
                        GPU.CopyDeviceToDevice(PredictedInput.DataPtr(),
                                               ResultPredicted.GetDeviceSlice(i * DeviceBatch, Intent.Write),
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
                            using (TorchTensor Penalty = Discriminators[i].PenalizeGradient(realInput, PredictedInput, penaltyLambda))
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

            prediction = ResultPredicted;
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

                    ResultPredicted.Dispose();

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
