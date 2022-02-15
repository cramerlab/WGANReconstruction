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
        private static extern IntPtr THSNN_ReconstructionWGANGenerator_apply_noise(Module.HType module, IntPtr fakeimages, IntPtr ctf);

        public TorchTensor ApplyNoise(TorchTensor fakeimages, TorchTensor ctf)
        {
            var res = THSNN_ReconstructionWGANGenerator_apply_noise(handle, fakeimages.Handle, ctf.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ReconstructionWGANGenerator_project(Module.HType module, IntPtr angles, double sigmashift);

        public TorchTensor Project(TorchTensor angles, double sigmashift)
        {
            var res = THSNN_ReconstructionWGANGenerator_project(handle, angles.Handle, sigmashift);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ReconstructionWGANGenerator_forward(Module.HType module, IntPtr angles, bool do_shift);

        public TorchTensor Forward(TorchTensor angles, bool do_shift)
        {
            var res = THSNN_ReconstructionWGANGenerator_forward(handle, angles.Handle, do_shift);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ReconstructionWGANGenerator_forward_new(Module.HType module, IntPtr angles, bool do_shift);

        public TorchTensor ForwardNew(TorchTensor angles, bool do_shift)
        {
            var res = THSNN_ReconstructionWGANGenerator_forward_new(handle, angles.Handle, do_shift);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ReconstructionWGANGenerator_forward_normalized(Module.HType module, IntPtr angles, IntPtr factor);

        public TorchTensor Forward_Normalized(TorchTensor angles, TorchTensor factor)
        {
            var res = THSNN_ReconstructionWGANGenerator_forward_normalized(handle, angles.Handle, factor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ReconstructionWGANGenerator_get_volume(Module.HType module);

        public TorchTensor Get_Volume()
        {
            var res = THSNN_ReconstructionWGANGenerator_get_volume(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
        
        [DllImport("LibTorchSharp")]
        private static extern void THSNN_ReconstructionWGANGenerator_apply_volume_masks(Module.HType module, IntPtr binaryMask, IntPtr maxMask, double multiplicator);

        public void ApplY_Volume_Mask(TorchTensor binaryMask, TorchTensor maxMask, double multiplicator)
        {
            THSNN_ReconstructionWGANGenerator_apply_volume_masks(handle, binaryMask.Handle, maxMask.Handle, multiplicator);
        }

    }

    public class ReconstructionWGANGeneratorSym : Module
    {
        internal ReconstructionWGANGeneratorSym(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern double THSNN_ReconstructionWGANGeneratorSym_clip_gradient(Module.HType module, double clip_Value);

        public double Clip_Gradients(double clip_Value)
        {
            return THSNN_ReconstructionWGANGeneratorSym_clip_gradient(handle, clip_Value);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ReconstructionWGANGeneratorSym_forward(Module.HType module, IntPtr angles, bool do_shift);

        public TorchTensor Forward(TorchTensor angles, bool do_shift)
        {
            var res = THSNN_ReconstructionWGANGeneratorSym_forward(handle, angles.Handle, do_shift);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ReconstructionWGANGeneratorSym_get_volume(Module.HType module);

        public TorchTensor Get_Volume()
        {
            var res = THSNN_ReconstructionWGANGeneratorSym_get_volume(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ReconstructionWGANGenerator_ctor(IntPtr volume, long boxsize, out IntPtr pBoxedModule);

        static public ReconstructionWGANGenerator ReconstructionWGANGenerator(TorchTensor volume, long boxsize)
        {
            var res = THSNN_ReconstructionWGANGenerator_ctor(volume.Handle, boxsize, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new ReconstructionWGANGenerator(res, boxedHandle);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ReconstructionWGANGeneratorSym_ctor(long boxsize, out IntPtr pBoxedModule);

        static public ReconstructionWGANGeneratorSym ReconstructionWGANGeneratorSym(long boxsize)
        {
            var res = THSNN_ReconstructionWGANGeneratorSym_ctor(boxsize, out var boxedHandle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new ReconstructionWGANGeneratorSym(res, boxedHandle);
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
            //var norm = NormalizeProjection(tensor);
            var res = THSNN_ReconstructionWGANDiscriminator_forward(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            //norm.Dispose();
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
        private static extern IntPtr THSNN_ReconstructionWGANDiscriminator_ctor(out IntPtr pBoxedModule, long boxsize, bool normalizeInput);

        static public ReconstructionWGANDiscriminator ReconstructionWGANDiscriminator(long boxsize, bool normalizeInput)
        {
            var res = THSNN_ReconstructionWGANDiscriminator_ctor(out var boxedHandle, boxsize, normalizeInput);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new ReconstructionWGANDiscriminator(res, boxedHandle);
        }
    }

    public class ReconstructionWGANSimpleCritic : Module
    {

        internal ReconstructionWGANSimpleCritic(IntPtr handle, IntPtr boxedHandle) : base(handle, boxedHandle) { }

        [DllImport("LibTorchSharp")]
        private static extern double THSNN_ReconstructionWGANSimpleCritic_clip_gradient(Module.HType module, double clip_Value);

        public double Clip_Gradients(double clip_Value)
        {
            return THSNN_ReconstructionWGANSimpleCritic_clip_gradient(handle, clip_Value);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ReconstructionWGANSimpleCritic_forward(Module.HType module, IntPtr tensor);

        public TorchTensor Forward(TorchTensor tensor)
        {
            //var norm = NormalizeProjection(tensor);
            var res = THSNN_ReconstructionWGANSimpleCritic_forward(handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            //norm.Dispose();
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
        private static extern void THSNN_ReconstructionWGANSimpleCritic_clipweights(Module.HType module, double clip);

        public void ClipWeights(double clip)
        {
            THSNN_ReconstructionWGANSimpleCritic_clipweights(handle, clip);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ReconstructionWGANSimpleCritic_penalizegradient(Module.HType module, IntPtr real, IntPtr fake, float lambda);

        public TorchTensor PenalizeGradient(TorchTensor real, TorchTensor fake, float lambda)
        {
            var res = THSNN_ReconstructionWGANSimpleCritic_penalizegradient(handle, real.Handle, fake.Handle, lambda);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
    }
    public static partial class Modules
    {
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSNN_ReconstructionWGANSimpleCritic_ctor(out IntPtr pBoxedModule, long boxsize, IntPtr image);

        static public ReconstructionWGANSimpleCritic ReconstructionWGANSimpleCritic(long boxsize, TorchTensor image)
        {
            var res = THSNN_ReconstructionWGANSimpleCritic_ctor(out var boxedHandle, boxsize, image.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new ReconstructionWGANSimpleCritic(res, boxedHandle);
        }
    }
}
