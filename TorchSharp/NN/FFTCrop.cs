using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{

    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ScaleVolume(IntPtr volume, int dim, int new_x, int new_y, int new_z);

        public static TorchTensor ScaleVolume(TorchTensor volume, int dim, int new_x, int new_y, int new_z)
        {
            var res = THSNN_ScaleVolume(volume.Handle, dim, new_x, new_y, new_z);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_FFTCrop(IntPtr fft_volume, int dim, int new_x, int new_y, int new_z);

        public static TorchTensor FFTCrop(TorchTensor fft_volume, int dim, int new_x, int new_y, int new_z)
        {
            var res = THSNN_FFTCrop(fft_volume.Handle, dim, new_x, new_y, new_z);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
}
