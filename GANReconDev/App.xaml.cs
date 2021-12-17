using System;
using System.Collections.Generic;
using System.Configuration;
using System.Data;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;

namespace ParticleWGANDev
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        public static string[] Args;
        void app_Startup(object sender, StartupEventArgs e)
        {
            
            // If no command line arguments were provided, don't process them if (e.Args.Length == 0) return;  
            if (e.Args.Length == 7)
            {
                int batchSize = int.Parse(e.Args[0]);
                decimal LearningRate = decimal.Parse(e.Args[1], CultureInfo.InvariantCulture.NumberFormat);
                decimal Reduction = decimal.Parse(e.Args[2], CultureInfo.InvariantCulture.NumberFormat);
                float Lambda = float.Parse(e.Args[3], CultureInfo.InvariantCulture.NumberFormat);
                int DiscIters = int.Parse(e.Args[4]); ;
                string LogFileName = e.Args[5];
                string OutDirectory = e.Args[6];
                ParticleWGANDev.MainWindow.settings = new(batchSize, LearningRate, Reduction, Lambda, DiscIters, LogFileName, OutDirectory);
            }
        }
    }
}
