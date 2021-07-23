using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{

    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_FFTCrop(IntPtr fft_volume, int new_fft_x, int new_fft_y, int new_fft_z);

        public static TorchTensor FFTCrop(TorchTensor fft_volume, int new_x, int new_y, int new_z)
        {
            var res = THSNN_FFTCrop(fft_volume.Handle, new_x, new_y, new_z);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_new_FFTCrop(IntPtr fft_volume, int dim, int new_x, int new_y, int new_z);

        public static TorchTensor newFFTCrop(TorchTensor fft_volume, int dim, int new_x, int new_y, int new_z)
        {
            var res = THSNN_new_FFTCrop(fft_volume.Handle, dim, new_x, new_y, new_z);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
}
