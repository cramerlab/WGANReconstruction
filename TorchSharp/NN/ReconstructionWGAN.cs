using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    public class ReconstructionWGANGenerator : Module
    {
        internal ReconstructionWGANGenerator(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern double THSNN_ReconstructionWGANGenerator_clip_gradient(Module.HType module, double clip_Value);

        public double Clip_Gradients(double clip_Value)
        {
            return THSNN_ReconstructionWGANGenerator_clip_gradient(handle, clip_Value);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ReconstructionWGANGenerator_forward_noise(Module.HType module, IntPtr crapcode, IntPtr fakeimages, IntPtr ctf);

        public TorchTensor ForwardNoise(TorchTensor crapcode, TorchTensor fakeimages, TorchTensor ctf)
        {
            var res = THSNN_ReconstructionWGANGenerator_forward_noise(handle, crapcode.Handle, fakeimages.Handle, ctf.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ReconstructionWGANGenerator_forward_particle(Module.HType module, IntPtr code, IntPtr angles, bool transform, double sigmashift);

        public TorchTensor ForwardParticle(TorchTensor code, TorchTensor angles, bool transform, double sigmashift)
        {
            var res = THSNN_ReconstructionWGANGenerator_forward_particle(handle, code.Handle, angles.Handle, transform, sigmashift);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ReconstructionWGANGenerator_ctor(IntPtr volume, long boxsize, long codelength, out IntPtr pBoxedModule);

        static public ReconstructionWGANGenerator ReconstructionWGANGenerator(TorchTensor volume, long boxsize, long codelength)
        {
            var res = THSNN_ReconstructionWGANGenerator_ctor(volume.Handle, boxsize, codelength, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new ReconstructionWGANGenerator(res, boxedHandle);
        }
    }

    public class ReconstructionWGANDiscriminator : Module
    {

        internal ReconstructionWGANDiscriminator(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }
        
        [DllImport("LibTorchSharp")]
        private static extern double THSNN_ReconstructionWGANDiscriminator_clip_gradient(Module.HType module, double clip_Value);

        public double Clip_Gradients(double clip_Value)
        {
            return THSNN_ReconstructionWGANDiscriminator_clip_gradient(handle, clip_Value);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ReconstructionWGANDiscriminator_forward(Module.HType module, IntPtr tensor);

        public TorchTensor Forward(TorchTensor tensor)
        {
            var norm = NormalizeProjection(tensor);
            var res = THSNN_ReconstructionWGANDiscriminator_forward(handle, norm.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public TorchTensor NormalizeProjection(TorchTensor t)
        {
            TorchTensor ret;
            using (TorchTensor mean = t.Mean(new long[] { 2, 3 }, true))
            using (TorchTensor std = t.Std(new long[] { 2, 3 }, true, true))
            {
                ret = (t - mean) / (std + 1e-6);
            }
            return ret;
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSNN_ReconstructionWGANDiscriminator_clipweights(Module.HType module, double clip);

        public void ClipWeights(double clip)
        {
            THSNN_ReconstructionWGANDiscriminator_clipweights(handle, clip);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ReconstructionWGANDiscriminator_penalizegradient(Module.HType module, IntPtr real, IntPtr fake, float lambda);

        public TorchTensor PenalizeGradient(TorchTensor real, TorchTensor fake, float lambda)
        {
            var res = THSNN_ReconstructionWGANDiscriminator_penalizegradient(handle, real.Handle, fake.Handle, lambda);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ReconstructionWGANDiscriminator_ctor(out IntPtr pBoxedModule);

        static public ReconstructionWGANDiscriminator ReconstructionWGANDiscriminator()
        {
            var res = THSNN_ReconstructionWGANDiscriminator_ctor(out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new ReconstructionWGANDiscriminator(res, boxedHandle);
        }
    }
}
