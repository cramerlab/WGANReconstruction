using System;
using System.Collections.Generic;
using System.Configuration;
using System.Data;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using CommandLine;

namespace CubeNetDev
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        public Options Options;

        private void Application_Startup(object sender, StartupEventArgs e)
        {
            Options = new Options();

            string ProgramFolder = System.Reflection.Assembly.GetEntryAssembly().Location;
            ProgramFolder = ProgramFolder.Substring(0, Math.Max(ProgramFolder.LastIndexOf('\\'), ProgramFolder.LastIndexOf('/')) + 1);

            if (!Debugger.IsAttached)
            {
                Parser.Default.ParseArguments<Options>(e.Args).WithParsed<Options>(opts => Options = opts);
                Options.WorkingDirectory = Environment.CurrentDirectory + "/";
            }
            else
            {
                Options.Mode = "infer";
                Options.OutputPath = "ribos";

                Options.WorkingDirectory = @"E:\cubenet\data";
                Options.VolumesPath = ".";
                Options.LabelsCoordsPath = ".";
                Options.LabelsSuffix = "_80S_denoised_flipx_clean";
                Options.DiagnosticsPath = "diag";

                Options.ParticleDiameter = 300 / 15;
                Options.MinimumVoxels = 15;

                Options.NLabels = 2;
                Options.LabelsRadius = 3;
                Options.OversampleBackground = 1;

                Options.NIterations = 10000;
                Options.WindowSize = 64;
                Options.BatchSize = 4;
                Options.PredictionThreshold = 0.1f;

                Options.OldModelName = "CubeNet20210414_212403.pt";

                Options.GPUNetwork = "0";
                Options.GPUPreprocess = 1;
            }
        }
    }
}
