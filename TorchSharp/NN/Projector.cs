using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class TorchProjector : Module
    {
        internal TorchProjector(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_Projector_Project(Module.HType module, IntPtr angles);

        public TorchTensor Project(TorchTensor angles)
        {
            var res = THSNN_Projector_Project(handle, angles.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_Projector_GetData(Module.HType module);

        public TorchTensor GetData()
        {
            var res = THSNN_Projector_GetData(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_Projector_GetCorrectedVolume(Module.HType module);

        public TorchTensor GetCorrectedVolume()
        {
            var res = THSNN_Projector_GetCorrectedVolume(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

    }
    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_Projector_ctor(IntPtr volume, int oversampling, out IntPtr outAsAnyModule);

        static public TorchProjector Projector(TorchTensor volume, int oversampling)
        {
            var res = THSNN_Projector_ctor(volume.Handle, oversampling, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchProjector(res, boxedHandle);
        }
    }
}
