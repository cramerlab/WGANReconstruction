using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{

    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_FFTCrop(IntPtr fft_volume, int new_fft_x, int new_fft_y, int new_fft_z);

        public static TorchTensor FFTCrop(TorchTensor fft_volume, int new_fft_x, int new_fft_y, int new_fft_z)
        {
            var res = THSNN_FFTCrop(fft_volume.Handle, new_fft_x, new_fft_y, new_fft_z);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
}
