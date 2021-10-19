using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

using Warp.Tools;
using TorchSharp.Tensor;
using TorchSharp.NN;
using static TorchSharp.NN.Modules;
using static TorchSharp.NN.Functions;
using static TorchSharp.NN.Losses;
using static TorchSharp.ScalarExtensionMethods;
using TorchSharp;

namespace Warp.NNModels
{
    public class ReconstructionWGAN
    {
        public readonly int2 BoxDimensions;
        public readonly int CodeLength;
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

        private TorchTensor[] TensorParticleCode;
        private TorchTensor[] TensorCrapCode;
        private TorchTensor[] TensorNoiseAdd;
        private TorchTensor[] TensorNoiseMul;

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

        //private float sigmaShift = 0.5 * (2f / BoxDimensions.X);
        private float sigmaShift = 0;

        private bool IsDisposed = false;

        private double GenVolumeBoost = 10;
        private double GenBoost = 1;
        private float generator_grad_clip_val = 10e6f;
        private float discriminator_grad_clip_val = 10e6f;
        private double lambdaOutsideMask = 10;

        public ReconstructionWGAN(int2 boxDimensions, int codeLength, int[] devices, int batchSize = 8, Image initialVolume = null)
        {
            Devices = devices;
            NDevices = Devices.Length;

            BatchSize = Math.Max(batchSize, NDevices);
            DeviceBatch = BatchSize / NDevices;
            if (BatchSize % NDevices != 0)
                throw new Exception("Batch size must be divisible by the number of devices.");

            BoxDimensions = boxDimensions;
            CodeLength = codeLength;

            Generators = new ReconstructionWGANGenerator[NDevices];
            Discriminators = new ReconstructionWGANDiscriminator[NDevices];

            TensorTrueImages = new TorchTensor[NDevices];
            TensorFakeImages = new TorchTensor[NDevices];
            TensorCTF = new TorchTensor[NDevices];
            TensorAngles = new TorchTensor[NDevices];

            TensorParticleCode = new TorchTensor[NDevices];
            TensorCrapCode = new TorchTensor[NDevices];
            TensorNoiseAdd = new TorchTensor[NDevices];
            TensorNoiseMul = new TorchTensor[NDevices];

            TensorOne = new TorchTensor[NDevices];
            TensorMinusOne = new TorchTensor[NDevices];

            TensorMask = new TorchTensor[NDevices];
            TensorVolumeMask = new TorchTensor[NDevices];

            

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                int DeviceID = Devices[i];


                Discriminators[i] = ReconstructionWGANDiscriminator();
                Discriminators[i].ToCuda(DeviceID);

                TensorTrueImages[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorFakeImages[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorCTF[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X / 2 + 1 }, DeviceType.CUDA, DeviceID);
                TensorAngles[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 3 }, DeviceType.CUDA, DeviceID);

                TensorParticleCode[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, CodeLength }, DeviceType.CUDA, DeviceID);
                TensorCrapCode[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, CodeLength }, DeviceType.CUDA, DeviceID);
                TensorNoiseAdd[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorNoiseMul[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);

                TensorOne[i] = Float32Tensor.Ones(new long[] { }, DeviceType.CUDA, DeviceID);
                TensorMinusOne[i] = Float32Tensor.Ones(new long[] { }, DeviceType.CUDA, DeviceID);
                {
                    GPU.CopyHostToDevice(new float[] { -1f }, TensorMinusOne[i].DataPtr(), 1);
                }

                TensorMask[i] = Float32Tensor.Ones(new long[] { 1, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                /*{
                    Image Mask = new Image(new int3(BoxDimensions.X, BoxDimensions.Y, 1));
                    Mask.Fill(1);
                    Mask.MaskSpherically(BoxDimensions.X / 2, BoxDimensions.X / 8, false);

                    GPU.CopyDeviceToDevice(Mask.GetDevice(Intent.Read), TensorMask[i].DataPtr(), Mask.ElementsReal);

                    Mask.Dispose();
                }*/
                TensorVolumeMask[i] = Float32Tensor.Ones(new long[] { 1, BoxDimensions.X, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                {
                    Image Mask = new Image(new int3(BoxDimensions.X, BoxDimensions.Y, BoxDimensions.X));
                    Mask.Fill(1);
                    Mask.MaskSpherically(BoxDimensions.X / 2+ 2*BoxDimensions.X / 8, BoxDimensions.X / 8, true);

                    GPU.CopyDeviceToDevice(Mask.GetDevice(Intent.Read), TensorVolumeMask[i].DataPtr(), Mask.ElementsReal);

                    Mask.Dispose();
                }

                TorchTensor volume = Float32Tensor.Zeros(new long[] { 1, 1, boxDimensions.X, boxDimensions.X, boxDimensions.X });
                //volume.RandomNInPlace(new long[] { 1, 1, boxDimensions.X, boxDimensions.X, boxDimensions.X });
                if (initialVolume != null)
                {
                    volume = TensorExtensionMethods.ToTorchTensor(initialVolume.GetHostContinuousCopy(), new long[] { 1, 1, boxDimensions.X, boxDimensions.X, boxDimensions.X }).ToDevice(DeviceType.CUDA, DeviceID);
                }
                Generators[i] = ReconstructionWGANGenerator(volume, BoxDimensions.X, codeLength);
                Generators[i].ToCuda(DeviceID);
            }, null);


            OptimizerGenVolume = Optimizer.Adam(Generators[0].GetParameters().Take(1), 0.01, 1e-4);
            OptimizerGen = Optimizer.Adam(Generators[0].GetParameters().Skip(1), 0.01, 1e-4);
            OptimizerDisc = Optimizer.Adam(Discriminators[0].GetParameters(), 0.01, 1e-4);

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
            var tensors = Generators[0].GetParameters();
            var tensorVolume = Generators[0].GetParameters()[0];

            var imageVolume = new Image(new int3(this.BoxDimensions.X));
            GPU.CopyDeviceToDevice(tensorVolume.DataPtr(), imageVolume.GetDevice(Intent.Write), imageVolume.ElementsReal);
            imageVolume.FreeDevice();
            return imageVolume;
        }
        public void TrainGeneratorParticle(float[] angles,
                                           Image imagesCTF,
                                           Image imagesReal,
                                           float learningRate,
                                           out Image prediction,
                                           out Image predictionNoisy,
                                           out float[] loss)
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

                //TensorParticleCode[i].RandomNInPlace(TensorParticleCode[i].Shape);
                TensorCrapCode[i].RandomNInPlace(TensorCrapCode[i].Shape);
                GPU.CopyHostToDevice(angles, TensorAngles[i].DataPtr(), DeviceBatch * 3);

                //TensorAngles[i].RandomNInPlace(TensorAngles[i].Shape);
                //TensorAngles[i] *= 2 * Math.PI;
                using (TorchTensor Prediction = Generators[i].ForwardParticle(TensorParticleCode[i], TensorAngles[i], true, sigmaShift))
                using (TorchTensor PredictionFT = Prediction.rfftn(new long[] { 2, 3 }))
                using (TorchTensor PredictionFTConv = PredictionFT.Mul(TensorCTF[i]))
                using (TorchTensor PredictionConv = PredictionFTConv.irfftn(new long[] { 2, 3 }))
                //using (TorchTensor PredictionIFFT = PredictionFT.irfftn(new long[] { 2, 3 }))
                //using (TorchTensor PredictionNoisy = Generators[i].ForwardNoise(TensorCrapCode[i], Prediction, TensorCTF[i]))
                using (TorchTensor PredictionNoisy = PredictionConv)
                //using (TorchTensor mean = PredictionNoisy.Mean(new long[] { 2, 3 }, true)) 
                //using (TorchTensor std = PredictionNoisy.Std(new long[] { 2, 3 }, true, true))
                //using (TorchTensor normalized = (PredictionNoisy - mean)/(std+1e-4))
                //using (TorchTensor PredictionMasked = PredictionNoisy.Mul(TensorMask[i]))
                using (TorchTensor IsItReal = Discriminators[i].Forward(PredictionNoisy))
                //using (TorchTensor IsItReal = TensorTrueImages[i] - PredictionMasked)
                //using (TorchTensor error = IsItReal.Pow(2))
                //using (TorchTensor Loss = error.Mean())
                using (TorchTensor Loss = ((-1)*IsItReal).Mean())
                {
                    /*{
                        Image PredictionFTImage = new Image(new int3(BoxDimensions.X, BoxDimensions.Y, DeviceBatch), true, false);
                        TorchTensor PredictionFTReal = PredictionFT.Abs();

                        GPU.CopyDeviceToDevice(PredictionFTReal.DataPtr(), PredictionFTImage.GetDevice(Intent.Write), PredictionFTImage.ElementsReal);
                        PredictionFTImage.WriteMRC(@"D:\GAN_recon_polcompl\PredictionFT.mrc", true);
                    }*/
                    /*{
                        Image PredictionFTConvImage = new Image(new int3(BoxDimensions.X, BoxDimensions.Y, DeviceBatch), true, false);
                        TorchTensor PredictionFTConvReal = PredictionFTConv.Abs();

                        GPU.CopyDeviceToDevice(PredictionFTConvReal.DataPtr(), PredictionFTConvImage.GetDevice(Intent.Write), PredictionFTConvImage.ElementsReal);
                        PredictionFTConvImage.WriteMRC(@"D:\GAN_recon_polcompl\PredictionFTConv.mrc", true);
                    }*/
                    GPU.CopyDeviceToDevice(Prediction.DataPtr(),
                                           ResultPredicted.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                           DeviceBatch * (int)BoxDimensions.Elements());
                    GPU.CopyDeviceToDevice(PredictionNoisy.DataPtr(),
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
                Generators[i].Clip_Gradients(generator_grad_clip_val);
            }, null);
            

            GatherGrads();

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
                                               out float[] lossFake)
        {
            OptimizerDisc.SetLearningRateAdam(learningRate);
            OptimizerDisc.ZeroGrad();

            SyncParams();
            ResultPredicted.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                Discriminators[i].Train();
                Discriminators[i].ZeroGrad();

                //Discriminators[i].ClipWeights(weightClip);

                GPU.CopyDeviceToDevice(imagesReal.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorTrueImages[i].DataPtr(),
                                       BoxDimensions.Elements() * DeviceBatch);
                GPU.CopyDeviceToDevice(imagesCTF.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorCTF[i].DataPtr(),
                                       DeviceBatch * BoxDimensions.ElementsFFT());

                float LossWasserstein = 0;

                //using (TorchTensor mean = TensorTrueImages[i].Mean(new long[] { 2, 3 }, true))
                //using (TorchTensor std = TensorTrueImages[i].Std(new long[] { 2, 3 }, true, true))
                //using (TorchTensor normalized = (TensorTrueImages[i] - mean) / (std + 1e-4))
                //using (TorchTensor masked = TensorTrueImages[i].Mul(TensorMask[i]))
                using (TorchTensor IsTrueReal = Discriminators[i].Forward(TensorTrueImages[i]))
                using (TorchTensor IsTrueRealNeg = IsTrueReal * (-1))
                using (TorchTensor LossReal = IsTrueRealNeg.Mean())
                {
                    if (i == 0)
                    {
                        GPU.CopyDeviceToHost(LossReal.DataPtr(), ResultLossDiscReal, 1);
                        LossWasserstein = ResultLossDiscReal[0];
                    }




                    //TensorParticleCode[i].RandomNInPlace(TensorParticleCode[i].Shape);
                    //TensorCrapCode[i].RandomNInPlace(TensorCrapCode[i].Shape);
                    //TensorAngles[i].RandomNInPlace(TensorAngles[i].Shape);
                    //TensorAngles[i] *= 2 * Math.PI;
                    GPU.CopyHostToDevice(angles, TensorAngles[i].DataPtr(), DeviceBatch * 3);
                    using (TorchTensor Prediction = Generators[i].ForwardParticle(TensorParticleCode[i], TensorAngles[i], true, sigmaShift))
                    using (TorchTensor PredictionFT = Prediction.rfftn(new long[] { 2, 3 }))
                    using (TorchTensor PredictionFTConv = PredictionFT.Mul(TensorCTF[i]))
                    using (TorchTensor PredictionConv = PredictionFTConv.irfftn(new long[] { 2, 3 }))
                    //using (TorchTensor PredictionNoisy = Generators[i].ForwardNoise(TensorCrapCode[i], Prediction, TensorCTF[i]))
                    using (TorchTensor PredictionNoisy = PredictionConv)
                    //using (TorchTensor PredictionNoisyMean = PredictionNoisy.Mean(new long[] { 2, 3 }, true))
                    //using (TorchTensor PredictionNoisyStd = PredictionNoisy.Std(new long[] { 2, 3 }, true, true))
                    //using (TorchTensor PredictionNoisyNormalized = (PredictionNoisy - PredictionNoisyMean) / (PredictionNoisyStd + 1e-4))
                    //using (TorchTensor PredictionMasked = PredictionNoisy)
                    //using (TorchTensor PredictionMasked = Prediction.Mul(TensorMask[i]))
                    using (TorchTensor PredictionDetached = PredictionNoisy.Detach())
                    using (TorchTensor IsFakeReal = Discriminators[i].Forward(PredictionDetached))
                    using (TorchTensor LossFake = IsFakeReal.Mean())
                    {
                        GPU.CopyDeviceToDevice(PredictionNoisy.DataPtr(),
                                               ResultPredicted.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                               DeviceBatch * (int)BoxDimensions.Elements());
                        if (i == 0)
                        {
                            GPU.CopyDeviceToHost(LossFake.DataPtr(), ResultLossDiscFake, 1);
                            LossWasserstein = LossWasserstein + ResultLossDiscFake[0];
                            ResultLoss[0] = LossWasserstein;
                        }


                        
                        using (TorchTensor Penalty = Discriminators[i].PenalizeGradient(TensorTrueImages[i], PredictionNoisy, penaltyLambda))
                        //using(TorchTensor added = LossFake+LossReal)
                        //using(TorchTensor wLoss = Penalty+ added)
                        {
                            LossReal.Backward();
                            LossFake.Backward();
                            Penalty.Backward();
                            //wLoss.Backward();
                        }
                    }
                }
                Discriminators[i].Clip_Gradients(discriminator_grad_clip_val);
            }, null);

            GatherGrads();

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
                //Generators[i].ToCuda(Devices[i]);

                Discriminators[i].Load(path + ".disc", DeviceType.CUDA, Devices[i]);
                //Discriminators[i].ToCuda(Devices[i]);
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
                        TensorCrapCode[i].Dispose();
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
