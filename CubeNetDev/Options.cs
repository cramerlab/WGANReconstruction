using CommandLine;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CubeNetDev
{
    public class Options
    {
        [Option("mode", Default = "train", HelpText = "train = train only; infer = infer only; both = train and infer on the same data")]
        public string Mode { get; set; }

        [Option("out", Default = "", HelpText = "Relative path to a folder that will contain the output.")]
        public string OutputPath { get; set; }

        [Option("volumes", Required = true, HelpText = "Relative path to a folder containing tomographic volumes.")]
        public string VolumesPath { get; set; }

        [Option("labels_star", Required = false, HelpText = "Relative path to a folder containing STAR files with normalized (values ranging from 0 to 1) particle coordinates. A labeled volume will be prepared based on these positions and --coords_star_radius.")]
        public string LabelsCoordsPath { get; set; }

        [Option("labels_star_radius", Default = 3, Required = false, HelpText = "Radius (in pixels) for the sphere that will be placed inside the labeled volumes at every particle's positions. Decrease for small particles to avoid overlaps.")]
        public int LabelsRadius { get; set; }

        [Option("labels_volume", Required = false, HelpText = "Relative path to a folder containing MRC volumes with the binary segmentations. Useful if you don't want to pick particles but rather segment some regions.")]
        public string LabelsVolumePath { get; set; }

        [Option("nlabels", Default = 2, Required = false, HelpText = "Number of distinct labels if using --labels_volume.")]
        public int NLabels { get; set; }

        [Option("oversample_background", Default = 1, Required = false, HelpText = "How many times should the background be sampled for each object sample?")]
        public int OversampleBackground { get; set; }

        [Option("labels_suffix", Default = "", HelpText = "Optional name suffix to relate volume name -> labels name.")]
        public string LabelsSuffix { get; set; }

        [Option("old_model", Default = "", HelpText = "Name of the folder with the pre-trained model. Leave empty to train a new one.")]
        public string OldModelName { get; set; }

        [Option("downsample", Default = 1, HelpText = "Downsample everything by this factor before training/inferrence. 1 = no change, >1 = smaller")]
        public int DownSample { get; set; }

        [Option("particle_diameter", Default = 20, HelpText = "Particle diameter in pixels that will be used to remove particles too close to the border.")]
        public int ParticleDiameter { get; set; }

        [Option("remove_noise", Default = 15, HelpText = "Connected components containing less than this many voxels will be removed from the segmentation.")]
        public int MinimumVoxels { get; set; }

        [Option("iterations", Default = 10000, HelpText = "Number of iterations.")]
        public int NIterations { get; set; }

        [Option("windowsize", Default = 64, HelpText = "If you have a lot of GPU memory and especially large particles, you can increase the network's window size in multiples of 32.")]
        public int WindowSize { get; set; }

        [Option("batchsize", Default = 4, HelpText = "Batch size for model training. Decrease if you run out of memory. The number of iterations will be adjusted automatically.")]
        public int BatchSize { get; set; }

        [Option("prediction_threshold", Default = 0.1f, HelpText = "Minimum certainty to consider a voxel labeled as not background.")]
        public float PredictionThreshold { get; set; }

        [Option("gpuid_network", Default = "0", HelpText = "GPU ID used for network training. If you want to split each batch between multiple GPUs, provide a comma-separated list of IDs, e.g. 0,1,2")]
        public string GPUNetwork { get; set; }

        [Option("gpuid_preprocess", Default = 1, HelpText = "GPU ID used for data preprocessing. Ideally not the GPU used for training")]
        public int GPUPreprocess { get; set; }

        [Option("diagnostics", Default = "", HelpText = "Relative path to a folder where diagnostic output will be saved, e.g. particle label volumes.")]
        public string DiagnosticsPath { get; set; }

        public string WorkingDirectory { get; set; }
    }
}
