using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class UNet3D : Module
    {
        internal UNet3D(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_UNet3D_forward(Module.HType module, IntPtr tensor);

        public TorchTensor Forward(TorchTensor tensor)
        {
            var res = THSNN_UNet3D_forward(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }

    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_UNet3D_ctor(long depth_block, long width_block, long input_channels, long final_channels, out IntPtr pBoxedModule);

        static public UNet3D UNet3D(long depth_block, long width_block, long input_channels, long final_channels)
        {
            var res = THSNN_UNet3D_ctor(depth_block, width_block, input_channels, final_channels, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new UNet3D(res, boxedHandle);
        }
    }
}
