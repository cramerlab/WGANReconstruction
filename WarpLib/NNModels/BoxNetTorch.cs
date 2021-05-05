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
    public class BoxNetTorch
    {
        public readonly int2 BoxDimensions;
        public readonly int BatchSize = 8;
        public readonly int DeviceBatch = 8;
        public readonly int[] Devices;
        public readonly int NDevices;

        private UNet2D[] UNetModel;

        private TorchTensor[] TensorSource;
        private TorchTensor[] TensorTarget;
        private TorchTensor[] TensorClassWeights;

        private Loss[] Loss;
        private Optimizer Optimizer;

        private Image ResultPredicted;
        private float[] ResultLoss = new float[1];

        private bool IsDisposed = false;

        public BoxNetTorch(int2 boxDimensions, float[] classWeights, int[] devices, int batchSize = 8)
        {
            Devices = devices;
            NDevices = Devices.Length;

            BatchSize = Math.Max(batchSize, NDevices);
            DeviceBatch = BatchSize / NDevices;
            if (BatchSize % NDevices != 0)
                throw new Exception("Batch size must be divisible by the number of devices.");

            BoxDimensions = boxDimensions;

            UNetModel = new UNet2D[NDevices];
            TensorSource = new TorchTensor[NDevices];
            TensorTarget = new TorchTensor[NDevices];
            TensorClassWeights = new TorchTensor[NDevices];

            Loss = new Loss[NDevices];
            if (classWeights.Length != 3)
                throw new Exception();

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                int DeviceID = Devices[i];

                UNetModel[i] = UNet2D(3, 1, 1, 3, 1, true, true);
                UNetModel[i].ToCuda(DeviceID);

                TensorSource[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 1, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);
                TensorTarget[i] = Float32Tensor.Zeros(new long[] { DeviceBatch, 3, BoxDimensions.Y, BoxDimensions.X }, DeviceType.CUDA, DeviceID);

                TensorClassWeights[i] = Float32Tensor.Zeros(new long[] { 3 }, DeviceType.CUDA, DeviceID);
                GPU.CopyHostToDevice(classWeights, TensorClassWeights[i].DataPtr(), 3);

                Loss[i] = CE(TensorClassWeights[i]);

            }, null);
            Optimizer = Optimizer.SGD(UNetModel[0].GetParameters(), 0.01, 0.9, false, 5e-4);

            ResultPredicted = new Image(IntPtr.Zero, new int3(BoxDimensions.X, BoxDimensions.Y, BatchSize));
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
                UNetModel[0].SynchronizeTo(UNetModel[i], Devices[i]);
        }

        private void GatherGrads()
        {
            for (int i = 1; i < NDevices; i++)
                UNetModel[0].GatherGrad(UNetModel[i]);
        }


        public void Predict(Image data, out Image prediction)
        {
            ScatterData(data, TensorSource);
            ResultPredicted.GetDevice(Intent.Write);

            Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                UNetModel[i].Eval();

                using (TorchTensor Prediction = UNetModel[i].Forward(TensorSource[i]))
                    GPU.CopyDeviceToDevice(Prediction.DataPtr(),
                                           ResultPredicted.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                           DeviceBatch * BoxDimensions.Elements());
            }, null);

            prediction = ResultPredicted;
        }

        public void Train(Image source,
                          Image target,
                          float learningRate,
                          bool needOutput,
                          out Image prediction,
                          out float[] loss)
        {
            GPU.CheckGPUExceptions();

            Optimizer.SetLearningRateSGD(learningRate);
            Optimizer.ZeroGrad();

            SyncParams();
            ResultPredicted.GetDevice(Intent.Write);

            //Helper.ForCPU(0, NDevices, NDevices, null, (i, threadID) =>
            {
                int i = 0;

                UNetModel[i].Train();
                UNetModel[i].ZeroGrad();

                GPU.CopyDeviceToDevice(source.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorSource[i].DataPtr(),
                                       DeviceBatch * (int)BoxDimensions.Elements());
                GPU.CopyDeviceToDevice(target.GetDeviceSlice(i * DeviceBatch, Intent.Read),
                                       TensorTarget[i].DataPtr(),
                                       DeviceBatch * (int)BoxDimensions.Elements() * 3);

                GPU.CheckGPUExceptions();

                using (TorchTensor TargetArgMax = TensorTarget[i].Argmax(1))
                using (TorchTensor Prediction = UNetModel[i].Forward(TensorSource[i]))
                using (TorchTensor PredictionLoss = Loss[i](Prediction, TargetArgMax))
                {
                    if (needOutput)
                    {
                        using (TorchTensor PredictionArgMax = Prediction.Argmax(1))
                        using (TorchTensor PredictionArgMaxFP = PredictionArgMax.ToType(ScalarType.Float32))
                        {
                            GPU.CopyDeviceToDevice(PredictionArgMaxFP.DataPtr(),
                                                   ResultPredicted.GetDeviceSlice(i * DeviceBatch, Intent.Write),
                                                   DeviceBatch * (int)BoxDimensions.Elements());
                        }
                    }

                    if (i == 0)
                        GPU.CopyDeviceToHost(PredictionLoss.DataPtr(), ResultLoss, 1);

                    PredictionLoss.Backward();
                }
            }//, null);

            GatherGrads();

            if (NDevices > 1)
                UNetModel[0].ScaleGrad(1f / NDevices);

            Optimizer.Step();

            prediction = ResultPredicted;
            loss = ResultLoss;
        }

        public void Save(string path)
        {
            Directory.CreateDirectory(Helper.PathToFolder(path));

            UNetModel[0].Save(path);
        }

        public void Load(string path)
        {
            for (int i = 0; i < NDevices; i++)
            {
                UNetModel[i].Load(path, DeviceType.CUDA, Devices[i]);
                //UNetModel[i].ToCuda(Devices[i]);
            }
        }

        ~BoxNetTorch()
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
                        TensorSource[i].Dispose();
                        TensorTarget[i].Dispose();
                        TensorClassWeights[i].Dispose();

                        UNetModel[i].Dispose();
                    }

                    Optimizer.Dispose();
                }
            }
        }
    }
}
