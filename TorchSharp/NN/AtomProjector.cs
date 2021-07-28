using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class AtomProjector : Module
    {
        internal AtomProjector(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_AtomProjector_ProjectToPlane(Module.HType module, IntPtr positions, IntPtr orientations, IntPtr shift);

        public TorchTensor ProjectToPlane(TorchTensor positions, TorchTensor orientations, TorchTensor shift)
        {
            var res = THSNN_AtomProjector_ProjectToPlane(handle, positions.Handle, orientations.Handle, shift.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_AtomProjector_RasterToCartesian(Module.HType module, IntPtr positions, IntPtr orientations, IntPtr shift);

        public TorchTensor RasterToCartesian(TorchTensor positions, TorchTensor orientations, TorchTensor shift)
        {
            var res = THSNN_AtomProjector_RasterToCartesian(handle, positions.Handle, orientations.Handle, shift.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ProjectAtomsToPlane(IntPtr intensities, IntPtr positions, IntPtr orientations, IntPtr shift, long sizeX, long sizeY, long sizeZ);

        public static TorchTensor ProjectAtomsToPlane(TorchTensor intensities, TorchTensor positions, TorchTensor orientations, TorchTensor shifts, int sizeX, int sizeY, int sizeZ)
        {
            var res = THSNN_ProjectAtomsToPlane(intensities.Handle, positions.Handle, orientations.Handle, shifts.Handle, sizeX, sizeY, sizeZ);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_RasterAtomsToCartesian(IntPtr intensities, IntPtr positions, IntPtr orientations, IntPtr shift, long sizeX, long sizeY, long sizeZ);

        public static TorchTensor RasterAtomsToCartesian(TorchTensor intensities, TorchTensor positions, TorchTensor orientations, TorchTensor shifts, int sizeX, int sizeY, int sizeZ)
        {
            var res = THSNN_RasterAtomsToCartesian(intensities.Handle, positions.Handle, orientations.Handle, shifts.Handle, sizeX, sizeY, sizeZ);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_AtomProjector_ctor(IntPtr intensities, int sizeX, int sizeY, int sizeZ, out IntPtr outAsAnyModule);

        static public AtomProjector AtomProjector(TorchTensor intensities, int sizeX, int sizeY, int sizeZ)
        {
            var res = THSNN_AtomProjector_ctor(intensities.Handle, sizeX, sizeY, sizeZ, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new AtomProjector(res, boxedHandle);
        }
    }
}
