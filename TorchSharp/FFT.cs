using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THS_RFFTN(IntPtr volume);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THS_IRFFTN(IntPtr fft_volume);

        public static TorchTensor RFFT(TorchTensor volume)
        {
            var res = THS_RFFTN(volume.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public static TorchTensor IRFFT(TorchTensor fft_volume)
        {
            var res = THS_IRFFTN(fft_volume.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
}
