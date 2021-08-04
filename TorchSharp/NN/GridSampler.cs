using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{

    public static partial class Modules
    {


        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ComplexGridSampling(IntPtr input, IntPtr grid, double max_r2);

        public static TorchTensor GridSample(TorchTensor input, TorchTensor grid, double max_r2)
        {
            var res = THSNN_ComplexGridSampling(input.Handle, grid.Handle, max_r2);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
}

