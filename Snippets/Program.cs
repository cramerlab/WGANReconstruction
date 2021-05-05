using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;
using Accord.Math.Optimization;
using System.Diagnostics;
using System.IO;
using System.Globalization;
using Warp.Sociology;
using Warp.Headers;
using Accord.MachineLearning;
using TorchSharp.Tensor;
using TorchSharp.NN;
using static TorchSharp.NN.Modules;
using static TorchSharp.NN.Functions;
using static TorchSharp.NN.Losses;
using TorchSharp;
using System.Runtime.InteropServices;

namespace Snippets
{
    class Program
    {
        private readonly static int _epochs = 10;
        private readonly static long _trainBatchSize = 64;
        private readonly static long _testBatchSize = 1000;
        private readonly static string _dataLocation = @"D:/Dev/TorchTest/MNIST";

        private readonly static int _logInterval = 10;

        static void Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.DefaultThreadCurrentUICulture = CultureInfo.InvariantCulture;

            // Test fiducial alignment
            if (false)
            {
                //float3[] Angles = Helper.ArrayOfFunction(i => new float3(0, (-40 + i * 2) * Helper.ToRad, 0), 41);
                float3[] Angles = Helper.ArrayOfFunction(i => new float3(0, (-57 + i * 3) * Helper.ToRad, 0), 40);
                float3[] AnglesExtended = Helper.ArrayOfFunction(i => new float3(0, (-50 + i * 2) * Helper.ToRad, 0), 51);

                if (false)
                {
                    Image Ref = Image.FromFile(@"D:\Dev\TiltSeriesAgain\run_it023_half1_class001.mrc");
                    //Image RefFlipped = Ref.AsFlippedX();
                    //Ref.Add(RefFlipped);
                    //RefFlipped = Ref.AsFlippedY();
                    //Ref.Add(RefFlipped);
                    //RefFlipped = Ref.AsFlippedZ();
                    //Ref.Add(RefFlipped);
                    Ref.TransformValues((x, y, z, v) => Math.Max(0, Math.Min(3, (9 - Math.Abs(z - Ref.Dims.Z / 2 - 0.0f * (x - Ref.Dims.X / 2))))) / 3f * v);
                    Ref.WriteMRC("d_ref.mrc", true);

                    Image ProjectionsRealspace = new Image(new int3(128 * 8, 128 * 8, Angles.Length));
                    Ref.RealspaceProject(Angles, ProjectionsRealspace, 8);
                    ProjectionsRealspace = ProjectionsRealspace.AsScaled(new int2(128)).AsPadded(new int2(80));

                    ProjectionsRealspace = Image.FromFile(@"D:\Dev\TiltSeriesAgain\TS_01.mrc.mrc").AsPadded(new int2(800)).AsScaled(new int2(100)).AsPadded(new int2(80));

                    ProjectionsRealspace.SubtractMeanGrid(new int2(1));
                    ProjectionsRealspace.Normalize();

                    //Image Determinism = Image.FromFile("d_projreal.mrc");
                    //Determinism.Subtract(ProjectionsRealspace);
                    //Determinism.Divide(ProjectionsRealspace);
                    //Debug.WriteLine(Math.Sqrt(MathHelper.Mean(Determinism.GetHostContinuousCopy().Select(v => v * v))));

                    ProjectionsRealspace.WriteMRC("d_projreal.mrc", true);

                    //GPU.DistortImagesAffine(ProjectionsRealspace.GetDevice(Intent.Read),
                    //                        new int2(ProjectionsRealspace.Dims),
                    //                        ProjectionsRealspace.GetDevice(Intent.Write),
                    //                        new int2(ProjectionsRealspace.Dims),
                    //                        Helper.ToInterleaved(new[] { new float2(0, 0) }),
                    //                        new[] { 1f, 0.3f, 0.2f, 1.2f },
                    //                        1);
                    //ProjectionsRealspace.WriteMRC("d_projrealdistorted.mrc", true);

                    //Image RecRealspace = new Image(new int3(128, 64, 64));
                    //RecRealspace.RealspaceBackproject(ProjectionsRealspace.AsScaled(new int2(64 * 8)), Angles, 8);
                    //RecRealspace.WriteMRC("d_recreal.mrc", true);

                    //ProjectionsRealspace = ProjectionsRealspace.AsPaddedClamped(new int2(ProjectionsRealspace.Dims) + 16);
                    //ProjectionsRealspace.Taper(8);

                    //Image RecSIRT = Image.ReconstructSIRT(ProjectionsRealspace, Angles, new int3(128, 80, 64), 8, 50);
                    //RecSIRT.WriteMRC("d_recsirt.mrc", true);

                    //Image ProjectionsExtended = new Image(new int3(80 * 8, 80 * 8, AnglesExtended.Length));
                    //RecSIRT.RealspaceProject(AnglesExtended, ProjectionsExtended, 8);
                    //ProjectionsExtended.AsScaled(new int2(80)).AsPadded(new int2(64)).WriteMRC("d_projectionsextended.mrc", true);

                    int[] SortedAngleIndices = Helper.AsSortedIndices(Angles, (a, b) => Math.Abs(a.Y).CompareTo(Math.Abs(b.Y)));


                    List<float[]> AlignedData = new List<float[]>(new[] { ProjectionsRealspace.GetHost(Intent.Read)[SortedAngleIndices[0]] });

                    int Dim = 80;
                    int Taper = 8;
                    int Super = 4;
                    int Depth = 40;

                    float3[] PlaneNormals = Angles.Select(a => Matrix3.Euler(a).Transposed() * float3.UnitZ).ToArray();
                    Image ProjFilter = new Image(new int3(Dim, Dim, Angles.Length), true);
                    float[][] ProjFilterData = ProjFilter.GetHost(Intent.Write);
                    for (int z = 0; z < Angles.Length; z++)
                        Helper.ForEachElementFT(new int2(ProjFilter.Dims), (x, y, xx, yy) =>
                        {
                            float Sum = 0;
                            float3 Pos = Matrix3.Euler(Angles[z]).Transposed() * new float3(xx, yy, 0);

                            for (int z2 = 0; z2 < Angles.Length; z2++)
                                Sum += Math.Max(0, 1 - Math.Abs(float3.Dot(Pos, PlaneNormals[z2])));

                            ProjFilterData[z][y * (ProjFilter.Dims.X / 2 + 1) + x] = 1 / Sum;
                        });
                    ProjFilter.WriteMRC("d_projfilter.mrc", true);

                    Image VolFilter = new Image(new int3(Dim, Dim, Depth), true);
                    float[][] VolFilterData = VolFilter.GetHost(Intent.Write);
                    Helper.ForEachElementFT(VolFilter.Dims, (x, y, z, xx, yy, zz, r) =>
                    {
                        float Sum = 0;
                        float3 Pos = new float3(xx, yy, zz);

                        for (int z2 = 0; z2 < Angles.Length; z2++)
                            Sum += Math.Max(0, 1 - Math.Abs(float3.Dot(Pos, PlaneNormals[z2])));

                        VolFilterData[z][y * (VolFilter.Dims.X / 2 + 1) + x] = Sum > 1 ? 1 : 0f;
                    });
                    VolFilter.WriteMRC("d_volfilter.mrc", true);

                    {
                        Image ProjectionsFT = ProjectionsRealspace.AsFFT();
                        ProjectionsFT.Multiply(ProjFilter);
                        ProjectionsRealspace = ProjectionsFT.AsIFFT(false, 0, true);

                        ProjectionsRealspace.WriteMRC("d_projfiltered.mrc", true);
                    }



                    Image MaskTaper = new Image(new int3(Dim, Dim, 1));
                    MaskTaper.Fill(1);
                    MaskTaper.Taper(10);

                    Image VolumeTaper = new Image(new int3(Dim, Dim, Depth));
                    VolumeTaper.Fill(1);
                    VolumeTaper.Taper(20);

                    if (false)
                    {
                        RandomNormal RandN = new RandomNormal(123);
                        int ioptim = 0;

                        Image Weights = new Image(new int3(Dim - 0, Dim - 0, Angles.Length));
                        Image WeightsVolume = new Image(new int3(Dim, Dim, Depth));
                        WeightsVolume.TransformValues((x, y, z, v) => 15 - Math.Abs(z - 15));
                        WeightsVolume.RealspaceProject(Angles, Weights, 1);
                        float[] WeightsMax = Weights.GetHost(Intent.Read).Select(a => 1f / MathHelper.Max(a)).ToArray();
                        Weights.Multiply(WeightsMax);

                        Weights.WriteMRC("d_weights.mrc", true);

                        WeightsVolume.Fill(1f);
                        WeightsVolume = WeightsVolume.AsPadded(new int3(Dim * 2, Dim, WeightsVolume.Dims.Z));
                        WeightsVolume.Add(-1f);
                        WeightsVolume.Multiply(-1f);
                        WeightsVolume.RealspaceProject(Angles, Weights, 1);

                        Weights.Binarize(0.5f);
                        Weights.Add(-1f);
                        Weights.Multiply(-1f);

                        WeightsVolume = new Image(new int3(Dim, Dim, Depth));
                        WeightsVolume.RealspaceBackproject(Weights, Angles, 1, false);
                        WeightsVolume.Binarize(Angles.Length / 2 - 0.5f);
                        WeightsVolume = FSC.MakeSoftMask(WeightsVolume, 0, 4);
                        WeightsVolume.WriteMRC("d_weightsvolume.mrc", true);

                        Image[] HalfProjections = Helper.ArrayOfFunction(i => new Image(new int3(Dim, Dim, Angles.Length / 2)), 2);
                        float3[][] HalfAngles = Helper.ArrayOfFunction(i => new float3[Angles.Length / 2], 2);
                        for (int i = 0; i < Angles.Length / 2; i++)
                        {
                            HalfProjections[0].GetHost(Intent.Write)[i] = ProjectionsRealspace.GetHost(Intent.Read)[i * 2 + 0].ToArray();
                            HalfProjections[1].GetHost(Intent.Write)[i] = ProjectionsRealspace.GetHost(Intent.Read)[i * 2 + 1].ToArray();

                            HalfAngles[0][i] = Angles[i * 2 + 0];
                            HalfAngles[1][i] = Angles[i * 2 + 1];
                        }

                        float[] InitialM = null;

                        Func<double[], double> Eval = input =>
                        {
                            Image[] Recs = new Image[2];

                            //for (int ihalf = 0; ihalf < 2; ihalf++)
                            //{
                            //    Image AlignedStackRaw = HalfProjections[ihalf].GetCopyGPU();
                            //    AlignedStackRaw.ShiftSlices(Helper.ArrayOfFunction(s => new float3((float)input[(s * 2 + ihalf) * 2 + 0], (float)input[(s * 2 + ihalf) * 2 + 1], 0), input.Length / 2 / 2));
                            //    //AlignedStackRaw.Normalize();

                            //    //Image AlignedStackSmooth = AlignedStackRaw.AsPaddedClamped(new int2(Dim + Taper * 2));
                            //    //AlignedStackRaw.Dispose();
                            //    //AlignedStackSmooth.Taper(Taper);
                            //    ////Image Residuals = new Image(AlignedStackSmooth.Dims);


                            //    //Image CurrentRec = Image.ReconstructSIRT(AlignedStackSmooth,
                            //    //                                         HalfAngles[ihalf],
                            //    //                                         new int3((Dim + Taper * 0) * 1, Dim + Taper * 0, Depth),
                            //    //                                         Super,
                            //    //                                         40,
                            //    //                                         null);

                            //    Image AlignedStackScaled = AlignedStackRaw.AsScaled(new int2(Dim * Super));
                            //    AlignedStackRaw.Dispose();
                            //    Image CurrentRec = new Image(new int3(Dim, Dim, Depth));
                            //    CurrentRec.RealspaceBackproject(AlignedStackScaled, HalfAngles[ihalf], Super);
                            //    AlignedStackScaled.Dispose();

                            //    CurrentRec.Multiply(VolumeTaper);

                            //    Image CurrentRecFT = CurrentRec.AsFFT(true);
                            //    CurrentRecFT.Multiply(VolFilter);
                            //    CurrentRec.Dispose();
                            //    CurrentRec = CurrentRecFT.AsIFFT(true, 0, true);
                            //    CurrentRecFT.Dispose();

                            //    Recs[ihalf] = CurrentRec;//.AsPadded(new int3(Dim / 2, Dim - 10, Depth));

                            //    //AlignedStackSmooth.Dispose();
                            //}

                            //float[] F = FSC.GetFSCNonCubic(Recs[0], Recs[1], 20);

                            //float[] RecData1 = Recs[0].GetHostContinuousCopy();
                            //Recs[0].Dispose();
                            //float[] RecData2 = Recs[1].GetHostContinuousCopy();
                            //Recs[1].Dispose();

                            Image AlignedStackRaw = ProjectionsRealspace.GetCopyGPU();
                            AlignedStackRaw.ShiftSlices(Helper.ArrayOfFunction(s => new float3((float)input[s * 2 + 0], (float)input[s * 2 + 1], 0), input.Length / 2));
                            //AlignedStackRaw.Normalize();

                            //Image AlignedStackSmooth = AlignedStackRaw.AsPaddedClamped(new int2(Dim + Taper * 2));
                            //AlignedStackRaw.Dispose();
                            //AlignedStackSmooth.Taper(Taper);
                            ////Image Residuals = new Image(AlignedStackSmooth.Dims);


                            //Image CurrentRec = Image.ReconstructSIRT(AlignedStackSmooth,
                            //                                         HalfAngles[ihalf],
                            //                                         new int3((Dim + Taper * 0) * 1, Dim + Taper * 0, Depth),
                            //                                         Super,
                            //                                         40,
                            //                                         null);

                            Image AlignedStackScaled = AlignedStackRaw.AsScaled(new int2(Dim * Super));
                            AlignedStackRaw.Dispose();
                            Image CurrentRec = new Image(new int3(Dim, Dim, Depth));
                            CurrentRec.RealspaceBackproject(AlignedStackScaled, Angles, Super);
                            AlignedStackScaled.Dispose();

                            CurrentRec.Multiply(VolumeTaper);

                            Image CurrentRecFT = CurrentRec.AsFFT(true);
                            CurrentRecFT.Multiply(VolFilter);
                            CurrentRec.Dispose();

                            float[] M = CurrentRecFT.AsAmplitudes1D();
                            float[] V = CurrentRecFT.AsAmplitudeVariance1D().Select(v => (float)Math.Sqrt(v)).ToArray();

                            CurrentRecFT.Dispose();

                            V = V.Select((v, i) => v / Math.Max(1e-10f, M[i])).ToArray();

                            //MathHelper.NormalizeInPlace(RecData1);
                            //MathHelper.NormalizeInPlace(RecData2);

                            //double Score = MathHelper.Subtract(RecData1, RecData2).Select(v => (double)v * v).Sum() / 100;

                            if (InitialM == null)
                                InitialM = M;

                            float M1 = M[1];
                            M = M.Select((v, i) => v / M1).ToArray();
                            M[0] = 1;
                            float3 Line = MathHelper.FitLineWeighted(Helper.ArrayOfFunction(i => new float3(i, M[i], 1f), M.Length));

                            double Score = Line.X;

                            //AlignedStackSmooth.Abs();
                            //AlignedStackSmooth.Max(1e-9f);
                            //Residuals.Divide(AlignedStackSmooth);

                            //Image ResidualsCropped = Residuals.AsPadded(new int2(Dim - 16));
                            //ResidualsCropped.Multiply(Weights);
                            //ResidualsCropped.Multiply(ResidualsCropped);
                            //Residuals.Dispose();

                            //double Score = MathHelper.StdDev(CurrentRec.GetHostContinuousCopy()) * 100;// Math.Sqrt(ResidualsCropped.GetHostContinuousCopy().Sum());
                            //ResidualsCropped.Dispose();
                            //CurrentRec.Dispose();

                            return Score * 100;
                        };

                        Func<double[], double[]> Grad = input =>
                        {
                            double[] Result = new double[input.Length];

                            if (ioptim++ >= 200)
                                return Result;

                            for (int i = 0; i < input.Length; i++)
                            {
                                double Delta = 0.02;

                                double[] InputPlus = input.ToList().ToArray();
                                InputPlus[i] += Delta;
                                double ScorePlus = Eval(InputPlus);

                                double[] InputMinus = input.ToList().ToArray();
                                InputMinus[i] -= Delta;
                                double ScoreMinus = Eval(InputMinus);

                                Result[i] = (ScorePlus - ScoreMinus) / (Delta * 2);
                            }

                            Debug.WriteLine(Eval(input));

                            return Result;
                        };

                        double[] Params = Helper.ArrayOfFunction(a => (double)RandN.NextSingle(0, 0.0005f), Angles.Length * 2);
                        BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(Params.Length, Eval, Grad);

                        Eval(Params);

                        Optimizer.Maximize(Params);

                        Debug.WriteLine(Eval(Params));
                    }

                    if (false)
                    {
                        int PlanForw = GPU.CreateFFTPlan(new int3(Dim - 10, Dim - 10, 1), 1);
                        int PlanBack = GPU.CreateIFFTPlan(new int3(Dim - 10, Dim - 10, 1), 1);

                        List<float2> AlignedShifts = new List<float2>();

                        for (int iangle = 0; iangle < SortedAngleIndices.Length - 1; iangle += 2)
                        {
                            Image AlignedStackRaw = new Image(AlignedData.ToArray(), new int3(Dim, Dim, AlignedData.Count));
                            Image AlignedStackSmooth = AlignedStackRaw.AsPaddedClamped(new int2(Dim + Taper * 2));
                            AlignedStackRaw.Dispose();
                            AlignedStackSmooth.Taper(Taper);
                            AlignedStackSmooth.WriteMRC("d_alignedstack.mrc", true);

                            Image CurrentRec = Image.ReconstructSIRT(AlignedStackSmooth,
                                                                     Helper.IndexedSubset(Angles, SortedAngleIndices.Take(iangle + 1).ToArray()).Select(a => a + new float3(0, 0f * Helper.ToRad, 0)).ToArray(),
                                                                     new int3((Dim + Taper * 2) * 2, Dim + Taper * 2, 48),
                                                                     Super,
                                                                     100);

                            //Image Determinism = Image.FromFile($"d_currentrec_{iangle:D2}.mrc");
                            //Determinism.Subtract(CurrentRec);
                            //Debug.WriteLine(Math.Sqrt(MathHelper.Mean(Determinism.GetHostContinuousCopy().Select(v => v * v))));

                            CurrentRec.WriteMRC($"d_currentrec_{iangle:D2}.mrc", true);
                            AlignedStackSmooth.Dispose();

                            for (int pn = 1; pn <= Math.Min(2, SortedAngleIndices.Length - 1 - iangle); pn++)
                            {
                                Image AlignmentTargetSuper = new Image(new int3(Dim * 2 * Super, Dim * 2 * Super, 1));
                                CurrentRec.RealspaceProject(new[] { Angles[SortedAngleIndices[iangle + pn]] + new float3(0, 0f * Helper.ToRad, 0) }, AlignmentTargetSuper, Super);

                                Image AlignmentTargetPadded = AlignmentTargetSuper.AsScaled(new int2(Dim * 2));
                                AlignmentTargetSuper.Dispose();
                                Image AlignmentTargetPadded2 = AlignmentTargetPadded.AsPadded(new int2(Dim));
                                AlignmentTargetPadded.Dispose();
                                AlignmentTargetPadded = AlignmentTargetPadded2;

                                Image AlignmentTargetFT = AlignmentTargetPadded.AsFFT();
                                AlignmentTargetFT.Multiply(ProjFilter);
                                AlignmentTargetPadded.Dispose();
                                AlignmentTargetPadded = AlignmentTargetFT.AsIFFT();
                                AlignmentTargetFT.Dispose();

                                Image AlignmentTarget = AlignmentTargetPadded.AsPadded(new int2(Dim - 10));
                                AlignmentTargetPadded.Dispose();

                                //AlignmentTarget.Bandpass(0.05f, 1.0f, false, 0.05f);
                                //AlignmentTarget.SubtractMeanGrid(new int2(1));

                                AlignmentTarget.Normalize();
                                AlignmentTarget.WriteMRC("d_alignmenttarget.mrc", true);

                                Image Template = new Image(ProjectionsRealspace.GetHost(Intent.Read)[SortedAngleIndices[iangle + pn]].ToArray(), new int3(Dim, Dim, 1));
                                Template.WriteMRC("d_template.mrc", true);

                                Image TemplateFT = Template.AsFFT();
                                TemplateFT.MultiplySlices(ProjFilter);
                                Template.Dispose();
                                Template = TemplateFT.AsIFFT();
                                TemplateFT.Dispose();

                                //Template.Bandpass(0.05f, 1.0f, false, 0.05f);
                                //Template.SubtractMeanGrid(new int2(1));

                                Template.Normalize();
                                Template.WriteMRC("d_templatefiltered.mrc", true);

                                Image TemplateDistorted = new Image(new int3(Dim - 10, Dim - 10, 1));
                                //Image TemplateDistortedFT = new Image(TemplateDistorted.Dims, true, true);

                                int ioptim = 0;

                                Func<double[], double> Eval = input =>
                                {
                                    float2 Shift = new float2((float)input[0], (float)input[1]);
                                    float4 Distortion = new float4((float)input[2] / 100,
                                                                   (float)input[3] / 100,
                                                                   (float)input[4] / 100,
                                                                   (float)input[5] / 100);

                                    GPU.DistortImagesAffine(Template.GetDevice(Intent.Read),
                                                            new int2(Template.Dims),
                                                            TemplateDistorted.GetDevice(Intent.Write),
                                                            new int2(TemplateDistorted.Dims),
                                                            Helper.ToInterleaved(new[] { Shift }),
                                                            Helper.ToInterleaved(new[] { Distortion }),
                                                            1);
                                    //TemplateDistorted.WriteMRC("d_templatedistorted.mrc", true);

                                    //GPU.FFT(TemplateDistorted.GetDevice(Intent.Read),
                                    //        TemplateDistortedFT.GetDevice(Intent.Write),
                                    //        TemplateDistorted.Dims.Slice(),
                                    //        1,
                                    //        PlanForw);

                                    //TemplateDistortedFT.MultiplySlices(ProjFilter);

                                    //GPU.IFFT(TemplateDistortedFT.GetDevice(Intent.Read),
                                    //         TemplateDistorted.GetDevice(Intent.Write),
                                    //         TemplateDistorted.Dims.Slice(),
                                    //         1,
                                    //         PlanBack,
                                    //         false);

                                    //Corr.Multiply(MaskTaper);
                                    TemplateDistorted.Normalize();
                                    TemplateDistorted.Multiply(AlignmentTarget);

                                    float Score = TemplateDistorted.GetHost(Intent.Read)[0].Sum();
                                    //Debug.WriteLine(Score);

                                    return Score;
                                };

                                Func<double[], double[]> Grad = input =>
                                {
                                    double[] Result = new double[input.Length];

                                    if (ioptim++ >= 20)
                                        return Result;

                                    for (int i = 0; i < 6; i++)
                                    {
                                        double Delta = i < 2 ? 0.02 : 0.05;

                                        double[] InputPlus = input.ToList().ToArray();
                                        InputPlus[i] += Delta;
                                        double ScorePlus = Eval(InputPlus);

                                        double[] InputMinus = input.ToList().ToArray();
                                        InputMinus[i] -= Delta;
                                        double ScoreMinus = Eval(InputMinus);

                                        Result[i] = (ScorePlus - ScoreMinus) / (Delta * 2);
                                    }

                                    Debug.WriteLine(Eval(input));

                                    return Result;
                                };

                                RandomNormal RandN = new RandomNormal(iangle + pn);
                                double[] Params = { 0,//RandN.NextSingle(0, 1),
                                        0,//RandN.NextSingle(0, 1),
                                        1 * 100f,
                                        0,
                                        0,
                                        1 * 100f };

                                BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(Params.Length, Eval, Grad);
                                Optimizer.Maximize(Params);

                                {
                                    float2 Shift = new float2((float)Params[0], (float)Params[1]);
                                    //Shift = new float2(1.5f, -0.7f);
                                    float4 Distortion = new float4((float)Params[2] / 100,
                                                                   (float)Params[3] / 100,
                                                                   (float)Params[4] / 100,
                                                                   (float)Params[5] / 100);
                                    Distortion = new float4(1, 0, 0, 1);

                                    GPU.DistortImagesAffine(ProjectionsRealspace.GetDeviceSlice(SortedAngleIndices[iangle + pn], Intent.Read),
                                                            new int2(Template.Dims),
                                                            Template.GetDevice(Intent.Write),
                                                            new int2(Template.Dims),
                                                            Helper.ToInterleaved(new[] { Shift }),
                                                            Helper.ToInterleaved(new[] { Distortion }),
                                                            1);

                                    Template.Normalize();

                                    //Image Determinism = Image.FromFile($"d_templatedistorted_{(iangle + pn):D2}.mrc");
                                    //Determinism.Subtract(Template);
                                    //Debug.WriteLine(Math.Sqrt(MathHelper.Mean(Determinism.GetHostContinuousCopy().Select(v => v * v))));

                                    Template.WriteMRC($"d_templatedistorted_{(iangle + pn):D2}.mrc", true);

                                    AlignedData.Add(Template.GetHostContinuousCopy());
                                    //AlignedData.Add(ProjectionsRealspace.GetHost(Intent.Read)[SortedAngleIndices[iangle + pn]]);
                                    AlignedShifts.Add(Shift);
                                }

                                //TemplateDistortedFT.Dispose();
                                TemplateDistorted.Dispose();
                                Template.Dispose();
                                AlignmentTarget.Dispose();
                            }

                            CurrentRec.Dispose();
                        }

                        Debug.WriteLine(MathHelper.Mean(AlignedShifts.Select(v => v.Length())));
                        Debug.WriteLine(AlignedShifts[0]);
                    }
                }

                if (true)
                {
                    int3 DimsExpanded = new int3(512, 512, 160);
                    int3 DimsVolume = new int3(512, 512, 160);
                    int Dim = 512;

                    Random Rand = new Random(123);
                    RandomNormal RandN = new RandomNormal(1234);

                    float3[] FiducialsExpanded = Helper.ArrayOfFunction(i => new float3((float)Rand.NextDouble() * DimsExpanded.X,
                                                                                        (float)Rand.NextDouble() * DimsExpanded.Y,
                                                                                        (float)Rand.NextDouble() * DimsExpanded.Z * 0.1f) - new float3(DimsExpanded.X / 2, DimsExpanded.Y / 2, 0), 10);

                    float3[][] TiltFiducials = new float3[Angles.Length][];
                    for (int t = 0; t < Angles.Length; t++)
                    {
                        Matrix3 M = Matrix3.Euler(Angles[t]).Transposed();
                        List<float3> Transformed = FiducialsExpanded.Select(v => M * v + (Angles[t].Y == 0 ? new float3(0) : new float3(RandN.NextSingle(0, 0.00002f), RandN.NextSingle(0, 0.00002f), 0))).ToList();
                        List<float3> Contained = Transformed.Where(v => Math.Abs(v.X) < Dim / 2 && Math.Abs(v.Y) < Dim / 2).ToList();
                        TiltFiducials[t] = Contained.ToArray();
                    }

                    float2[][] TiltFiducials2D = TiltFiducials.Select(a => a.Select(v => new float2(v)).ToArray()).ToArray();

                    int[] AngleIndicesSorted = Helper.AsSortedIndices(Angles, (a, b) => Math.Abs(a.Y).CompareTo(Math.Abs(b.Y)));
                    int[] CurrentAngleIndices = new int[0];

                    int ZeroTiltID = AngleIndicesSorted[0];
                    int NFiducials = TiltFiducials[ZeroTiltID].Length;

                    float[][] FidualTiltMasks = Helper.ArrayOfFunction(f => Helper.ArrayOfConstant(0.2f, Angles.Length), NFiducials);
                    float2[] TiltOffsets = Helper.ArrayOfFunction(i => new float2(RandN.NextSingle(0, 4f), RandN.NextSingle(0, 4f)), Angles.Length);
                    TiltOffsets[ZeroTiltID] = new float2(0);

                    {
                        Image Viz = new Image(new int3(Dim, Dim, Angles.Length));
                        float[][] VizData = Viz.GetHost(Intent.Read);
                        for (int a = 0; a < Angles.Length; a++)
                        {
                            foreach (var f in TiltFiducials[a])
                            {
                                int2 Coords = new int2(new float2(f) + Dim / 2 + TiltOffsets[a]);
                                if (Coords.X >= 0 && Coords.X < Dim && Coords.Y >= 0 && Coords.Y < Dim)
                                    VizData[a][Coords.Y * Dim + Coords.X] = 1;
                            }
                        }

                        Viz.WriteMRC("d_tiltviz.mrc", true);
                    }

                    int ioptim = 20;
                    bool OptimizeOffsets = true;
                    bool OptimizeFiducials = true;

                    Func<float3, float[], float> EvalOneFiducial = (pos, mask) =>
                    {
                        float Sum = 0;

                        foreach (var t in CurrentAngleIndices)
                        {
                            float2 Transformed = new float2(Matrix3.Euler(Angles[t]).Transposed() * pos) + TiltOffsets[t];

                            float Closest = float.MaxValue;
                            for (int f2 = 0; f2 < TiltFiducials2D[t].Length; f2++)
                            {
                                float Dist = (Transformed - TiltFiducials2D[t][f2]).Length();
                                Closest = Math.Min(Dist, Closest);
                            }

                            Sum += Closest * mask[t];
                        }

                        return Sum;
                    };

                    Func<float3, float[]> EvalOneFiducialPerTilt = pos =>
                    {
                        float[] Result = new float[CurrentAngleIndices.Length];

                        int i = 0;
                        foreach (var t in CurrentAngleIndices)
                        {
                            float2 Transformed = new float2(Matrix3.Euler(Angles[t]).Transposed() * pos) + TiltOffsets[t];

                            float Closest = float.MaxValue;
                            for (int f2 = 0; f2 < TiltFiducials2D[t].Length; f2++)
                            {
                                float Dist = (Transformed - TiltFiducials2D[t][f2]).Length();
                                Closest = Math.Min(Dist, Closest);
                            }

                            Result[i++] = Closest;
                        }

                        return Result;
                    };

                    Func<double[], int, double> EvalOneTilt = (input, t) =>
                    {
                        float3[] CurrentFiducials = Helper.FromInterleaved3(input.Take(NFiducials * 3).Select(v => (float)v).ToArray());
                        TiltOffsets = Helper.FromInterleaved2(input.Skip(NFiducials * 3).Select(v => (float)v).ToArray());

                        double SumDistance = 0;

                        for (int f1 = 0; f1 < CurrentFiducials.Length; f1++)
                        {
                            float2 Transformed = new float2(Matrix3.Euler(Angles[t]).Transposed() * CurrentFiducials[f1]) + TiltOffsets[t];

                            float Closest = float.MaxValue;
                            for (int f2 = 0; f2 < TiltFiducials2D[t].Length; f2++)
                            {
                                float Dist = (Transformed - TiltFiducials2D[t][f2]).Length();
                                Closest = Math.Min(Dist, Closest);
                            }

                            SumDistance += Closest * FidualTiltMasks[f1][t];
                        }

                        return SumDistance;
                    };

                    Func<double[], double> Eval = input =>
                    {
                        float3[] CurrentFiducials = Helper.FromInterleaved3(input.Take(NFiducials * 3).Select(v => (float)v).ToArray());
                        TiltOffsets = Helper.FromInterleaved2(input.Skip(NFiducials * 3).Select(v => (float)v).ToArray());

                        double SumDistance = 0;

                        for (int f1 = 0; f1 < CurrentFiducials.Length; f1++)
                            SumDistance += EvalOneFiducial(CurrentFiducials[f1], FidualTiltMasks[f1]);

                        return SumDistance;
                    };

                    Func<double[], double[]> Grad = input =>
                    {
                        double[] Result = new double[input.Length];
                        if (ioptim-- <= 0)
                            return Result;

                        Eval(input);

                        int Offset = 0;

                        if (OptimizeFiducials)
                            for (int f = 0; f < NFiducials; f++)
                            {
                                float Delta = 0.0001f;
                                float3 Pos = new float3((float)input[f * 3 + 0],
                                                        (float)input[f * 3 + 1],
                                                        (float)input[f * 3 + 2]);

                                float3[] Deltas = { float3.UnitX, float3.UnitY, float3.UnitZ };

                                for (int i = 0; i < 3; i++)
                                {
                                    double ScorePlus = EvalOneFiducial(Pos + Deltas[i] * Delta, FidualTiltMasks[f]);
                                    double ScoreMinus = EvalOneFiducial(Pos - Deltas[i] * Delta, FidualTiltMasks[f]);

                                    Result[f * 3 + i] = (ScorePlus - ScoreMinus) / (Delta * 2);
                                }
                            }

                        Offset += NFiducials * 3;

                        if (OptimizeOffsets)
                            for (int t = 0; t < TiltOffsets.Length * 2; t++)
                            {
                                if (t / 2 == ZeroTiltID || !CurrentAngleIndices.Contains(t / 2))
                                    continue;

                                double Delta = 0.0001;

                                double[] InputPlus = input.ToList().ToArray();
                                InputPlus[Offset + t] += Delta;
                                double ScorePlus = EvalOneTilt(InputPlus, t / 2);

                                double[] InputMinus = input.ToList().ToArray();
                                InputMinus[Offset + t] -= Delta;
                                double ScoreMinus = EvalOneTilt(InputMinus, t / 2);

                                Result[Offset + t] = (ScorePlus - ScoreMinus) / (Delta * 2);
                            }

                        Debug.WriteLine(Eval(input));

                        return Result;
                    };

                    double[] Params = Helper.ToInterleaved(TiltFiducials[ZeroTiltID].Select(v => new float3(v.X, v.Y, 0)).ToArray()).Select(v => (double)v).ToArray();
                    Params = Helper.Combine(Params, Helper.ToInterleaved(TiltOffsets).Select(v => (double)v).ToArray());
                    BroydenFletcherGoldfarbShanno Optimizer = new BroydenFletcherGoldfarbShanno(Params.Length, Eval, Grad);

                    for (int nt = 40; nt < Angles.Length + 4; nt += 4)
                    {
                        for (int repeat = 0; repeat < 2; repeat++)
                        {
                            CurrentAngleIndices = AngleIndicesSorted.Take(nt).ToArray();

                            Debug.WriteLine(Eval(Params));

                            float3[] CurrentFiducials = Helper.FromInterleaved3(Params.Take(NFiducials * 3).Select(v => (float)v).ToArray());
                            float[] Deviations = CurrentFiducials.Select((f, i) => EvalOneFiducial(f, FidualTiltMasks[i])).ToArray();
                            float2 MeanStd = MathHelper.MeanAndStd(Deviations);

                            for (int f = 0; f < NFiducials; f++)
                            {
                                //if (Deviations[f] - MeanStd.X > MeanStd.Y * 2)
                                {
                                    float3 Pos = CurrentFiducials[f];
                                    float LowestDeviation = EvalOneFiducial(Pos, FidualTiltMasks[f]);
                                    for (decimal z = -(decimal)DimsVolume.Z / 2; z < (decimal)DimsVolume.Z / 2; z += 0.02M)
                                    {
                                        float CurrentDeviation = EvalOneFiducial(new float3(Pos.X, Pos.Y, (float)z), FidualTiltMasks[f]);
                                        if (CurrentDeviation < LowestDeviation)
                                        {
                                            LowestDeviation = CurrentDeviation;
                                            Params[f * 3 + 2] = (double)z;
                                        }
                                    }
                                }
                            }

                            Debug.WriteLine(Eval(Params));

                            ioptim = 150;
                            OptimizeFiducials = false;
                            Optimizer.Minimize(Params);

                            CurrentFiducials = Helper.FromInterleaved3(Params.Take(NFiducials * 3).Select(v => (float)v).ToArray());

                            float[][] TiltDeviations = CurrentFiducials.Select(f => EvalOneFiducialPerTilt(f)).ToArray();
                            for (int it = 0; it < Math.Min(nt, Angles.Length); it++)
                            {
                                float[] CurrentTiltDeviations = TiltDeviations.Select(a => (float)Math.Sqrt(a[it])).ToArray();
                                MeanStd = MathHelper.MeanAndStd(CurrentTiltDeviations);
                                MeanStd.Y = Math.Max(0.05f, MeanStd.Y);

                                CurrentTiltDeviations = CurrentTiltDeviations.Select(v => Math.Max(0, (v - MeanStd.X) / MeanStd.Y)).ToArray();
                                CurrentTiltDeviations = CurrentTiltDeviations.Select(v => Math.Max(1e-6f, (float)Math.Exp(-(v * v) / 2))).ToArray();

                                int t = CurrentAngleIndices[it];
                                for (int f = 0; f < NFiducials; f++)
                                    FidualTiltMasks[f][t] = CurrentTiltDeviations[f];
                            }
                        }
                    }

                    {
                        float3[] CurrentFiducials = Helper.FromInterleaved3(Params.Take(NFiducials * 3).Select(v => (float)v).ToArray());
                        List<float> Deviations = new List<float>();
                        for (int f1 = 0; f1 < CurrentFiducials.Length; f1++)
                        {
                            Deviations.Add((CurrentFiducials[f1] - TiltFiducials[ZeroTiltID][f1]).Length());
                        }

                        Console.WriteLine(Deviations.Average());
                    }
                }
            }

            // Test TensorFlow memory deallocation
            if (false)
            {
                Debug.WriteLine(GPU.GetFreeMemory(0));

                NoiseNet3D Net = new NoiseNet3D(@"D:\Dev\Noise2Map\Noise2Map\bin\Debug\noisenet3d_64", new int3(64), 1, 4, true);

                Debug.WriteLine(GPU.GetFreeMemory(0));

                Net.Dispose();
                TFHelper.TF_FreeAllMemory();

                Net = new NoiseNet3D(@"D:\Dev\Noise2Map\Noise2Map\bin\Debug\noisenet3d_64", new int3(64), 1, 4, true);

                Debug.WriteLine(GPU.GetFreeMemory(0));

                Net.Dispose();
                TFHelper.TF_FreeAllMemory();
            }

            // Measure the Apoferritin Quantum Efficiency
            if (false)
            {
                string[] Suffixes = { "counting", "superres" };
                string InputFolderPath = @"H:\20190604_ApoF_combined\";
                int DimRec = -1;
                int DimOri = -1;
                float AngPixOri = 1.27f;
                float AngPix = 1.27f;

                int SubsetSize = 5000;
                int NSubsets = 50;

                Image Mask = Image.FromFile(InputFolderPath + "mask.mrc");

                Image Reference = Image.FromFile(InputFolderPath + "Refine3D/job002/run_half1_class001_unfil.mrc");
                Reference.Add(Image.FromFile(InputFolderPath + "Refine3D/job002/run_half2_class001_unfil.mrc"));
                Reference.Multiply(Mask);

                Projector RefProjector = new Projector(Reference, 2);

                foreach (var suffix in Suffixes)
                {
                    Star TableParticleMetadata = new Star(InputFolderPath + $"refined_{suffix}.star");

                    #region Read STAR

                    float3[] ParticleAngles = TableParticleMetadata.GetRelionAngles().Select(a => a * Helper.ToRad).ToArray();
                    float3[] ParticleShifts = TableParticleMetadata.GetRelionOffsets();

                    CTF[] ParticleCTFParams = TableParticleMetadata.GetRelionCTF();
                    {
                        float MeanNorm = MathHelper.Mean(ParticleCTFParams.Select(p => (float)p.Scale));
                        for (int p = 0; p < ParticleCTFParams.Length; p++)
                            ParticleCTFParams[p].Scale /= (decimal)MeanNorm;
                    }

                    int[] ParticleSubset = TableParticleMetadata.GetColumn("rlnRandomSubset").Select(v => int.Parse(v) - 1).ToArray();

                    string[] ParticleNames = TableParticleMetadata.GetColumn("rlnImageName");
                    string[] UniqueMicrographs = Helper.GetUniqueElements(ParticleNames.Select(s => s.Substring(s.IndexOf('@') + 1))).ToArray();

                    Console.WriteLine("Done.\n");

                    #endregion

                    #region Load data

                    int NParticles = TableParticleMetadata.RowCount;
                    int NThreads = 4;

                    //float[][] RawParticles = new float[NParticles][];
                    //float[][] RawCTFs = new float[NParticles][];
                    //Matrix3[] RawRotations = new Matrix3[NParticles];
                    //Matrix3[] SymmetryMatrices = new Symmetry("O").GetRotationMatrices();
                    double[][] AB = Helper.ArrayOfFunction(i => new double[new int3(DimOri).Slice().ElementsFFT()], NThreads);// new float[NParticles][];
                    double[][] A2 = Helper.ArrayOfFunction(i => new double[new int3(DimOri).Slice().ElementsFFT()], NThreads);
                    double[][] B2 = Helper.ArrayOfFunction(i => new double[new int3(DimOri).Slice().ElementsFFT()], NThreads);
                    int NDone = 0;

                    Helper.ForCPU(0, UniqueMicrographs.Length, NThreads, threadID => GPU.SetDevice(0),
                        (imic, threadID) =>
                        {
                            int[] RowIndices = Helper.GetIndicesOf(ParticleNames, (s) => s.Substring(s.IndexOf('@') + 1) == UniqueMicrographs[imic]);
                            string StackPath = InputFolderPath + ParticleNames[RowIndices[0]].Substring(ParticleNames[RowIndices[0]].IndexOf('@') + 1);

                            if (!File.Exists(StackPath))
                                throw new Exception($"No data found for {UniqueMicrographs[imic]}!");

                            Image OriginalStack = Image.FromFile(StackPath);

                            //lock (TableParticleMetadata)
                            //{
                            //    if (DimRec <= 0)   // Figure out dimensions using the first stack
                            //    {
                            //        DimOri = OriginalStack.Dims.X;
                            //        DimRec = (int)Math.Round(DimOri * AngPixOri / AngPix / 2) * 2;
                            //        AngPix = (float)DimOri / DimRec * AngPixOri;   // Adjust pixel size to match rounded box size
                            //    }
                            //}

                            int[] SliceIndices = Helper.IndexedSubset(ParticleNames, RowIndices).Select(s => int.Parse(s.Split(new[] { '@' })[0]) - 1).ToArray();

                            float3[] MicShifts = Helper.IndexedSubset(ParticleShifts, RowIndices);

                            Image OriginalStackFT = OriginalStack.AsFFT();
                            OriginalStack.Dispose();

                            float[][] RelevantStackData = Helper.IndexedSubset(OriginalStackFT.GetHost(Intent.Read), SliceIndices);
                            Image RelevantStack = new Image(RelevantStackData, new int3(DimOri, DimOri, SliceIndices.Length), true, true);
                            OriginalStackFT.Dispose();

                            RelevantStack.Multiply(1f / RelevantStack.Dims.ElementsSlice());

                            RelevantStack.ShiftSlices(MicShifts.Select(v => new float3(v.X + DimOri / 2, v.Y + DimOri / 2, 0)).ToArray());  // Shift and de-center

                            Image RelevantStackScaled = RelevantStack.AsPadded(new int2(DimRec));
                            RelevantStack.Dispose();
                            //RelevantStackScaled.AsIFFT().WriteMRC("d_stackscaled.mrc", true);

                            #region Create CTF

                            Image CTFCoords = CTF.GetCTFCoords(DimRec, DimRec);
                            CTFStruct[] CTFParams = new CTFStruct[RowIndices.Length];
                            for (int p = 0; p < RowIndices.Length; p++)
                            {
                                int R = RowIndices[p];
                                ParticleCTFParams[R].PixelSize = (decimal)AngPix;
                                CTFParams[p] = ParticleCTFParams[R].ToStruct();
                            }

                            Image StackCTF = new Image(new int3(DimRec, DimRec, RowIndices.Length), true);
                            GPU.CreateCTF(StackCTF.GetDevice(Intent.Write),
                                      CTFCoords.GetDevice(Intent.Read),
                                      IntPtr.Zero,
                                      (uint)CTFCoords.ElementsComplex,
                                      CTFParams,
                                      false,
                                      (uint)CTFParams.Length);
                            //StackCTF.WriteMRC("d_ctf.mrc", true);
                            CTFCoords.Dispose();

                            #endregion

                            //RelevantStackScaled.Multiply(StackCTF);
                            //RelevantStackScaled.FreeDevice();
                            //StackCTF.Multiply(StackCTF);
                            //StackCTF.FreeDevice();

                            RelevantStackScaled.Multiply(StackCTF);

                            Image RefProjections = RefProjector.Project(new int2(DimRec), RowIndices.Select(p => ParticleAngles[p]).ToArray());
                            RefProjections.Multiply(StackCTF);
                            RefProjections.Multiply(StackCTF);

                            for (int p = 0; p < RowIndices.Length; p++)
                            {
                                float[] ParticleData = RelevantStackScaled.GetHost(Intent.Read)[p];
                                float[] ProjData = RefProjections.GetHost(Intent.Read)[p];

                                float[] ParticleAB = new float[StackCTF.ElementsSliceReal];
                                float[] ParticleA2 = new float[StackCTF.ElementsSliceReal];
                                float[] ParticleB2 = new float[StackCTF.ElementsSliceReal];

                                for (int i = 0; i < ParticleAB.Length; i++)
                                {
                                    AB[threadID][i] += ParticleData[i * 2 + 0] * ProjData[i * 2 + 0] + ParticleData[i * 2 + 1] * ProjData[i * 2 + 1];
                                    A2[threadID][i] += ParticleData[i * 2 + 0] * ParticleData[i * 2 + 0] + ParticleData[i * 2 + 1] * ParticleData[i * 2 + 1];
                                    B2[threadID][i] += ProjData[i * 2 + 0] * ProjData[i * 2 + 0] + ProjData[i * 2 + 1] * ProjData[i * 2 + 1];
                                }

                                //AB[RowIndices[p]] = ParticleAB;
                                //A2[RowIndices[p]] = ParticleA2;
                                //B2[RowIndices[p]] = ParticleB2;

                                //RawParticles[RowIndices[p]] = RelevantStackScaled.GetHost(Intent.Read)[p];
                                //RawCTFs[RowIndices[p]] = StackCTF.GetHost(Intent.Read)[p];
                                //RawRotations[RowIndices[p]] = Matrix3.Euler(ParticleAngles[RowIndices[p]]);
                            }

                            RefProjections.Dispose();

                            RelevantStackScaled.Dispose();
                            StackCTF.Dispose();
                        }, null);

                    #endregion

                    float[] AB2D = new float[AB[0].Length];
                    float[] A22D = new float[AB[0].Length];
                    float[] B22D = new float[AB[0].Length];
                    for (int p = 0; p < AB.Length; p++)
                    {
                        for (int i = 0; i < AB2D.Length; i++)
                        {
                            AB2D[i] += (float)AB[p][i];
                            A22D[i] += (float)A2[p][i];
                            B22D[i] += (float)B2[p][i];
                        }
                    }

                    float[] AB1D = new float[DimRec / 2];
                    float[] A21D = new float[DimRec / 2];
                    float[] B21D = new float[DimRec / 2];
                    Helper.ForEachElementFT(new int2(DimRec), (x, y, xx, yy, r, angle) =>
                    {
                        if ((int)r < AB1D.Length)
                        {
                            float W1 = r - (int)r;
                            float W0 = 1 - W1;

                            AB1D[(int)r] += AB2D[y * (DimRec / 2 + 1) + x] * W0;
                            A21D[(int)r] += A22D[y * (DimRec / 2 + 1) + x] * W0;
                            B21D[(int)r] += B22D[y * (DimRec / 2 + 1) + x] * W0;

                            if (r + 1 < AB1D.Length)
                            {
                                AB1D[(int)r + 1] += AB2D[y * (DimRec / 2 + 1) + x] * W1;
                                A21D[(int)r + 1] += A22D[y * (DimRec / 2 + 1) + x] * W1;
                                B21D[(int)r + 1] += B22D[y * (DimRec / 2 + 1) + x] * W1;
                            }
                        }
                    });

                    float[] FSC1D = new float[AB1D.Length];
                    for (int i = 0; i < FSC1D.Length; i++)
                    {
                        FSC1D[i] = AB1D[i] / (float)Math.Sqrt(A21D[i] * B21D[i]);
                        Console.WriteLine(FSC1D[i]);
                    }

                    Console.WriteLine("");
                    Console.WriteLine("");

                    //int[][] HalfIndices = Helper.ArrayOfFunction(ihalf => Helper.ArrayOfSequence(0, TableParticleMetadata.RowCount, 1).Where(i => ParticleSubset[i] == ihalf).ToArray(), 2);

                    //float[][] FSCs = new float[NSubsets][];

                    //for (int isubset = 0; isubset < NSubsets; isubset++)
                    //{
                    //    Image[] Reconstructions = new Image[2];

                    //    for (int ihalf = 0; ihalf < 2; ihalf++)
                    //    {
                    //        int[] Indices = Helper.RandomSubset(HalfIndices[ihalf], SubsetSize, 123 + isubset);

                    //        Projector Reconstructor = new Projector(new int3(DimRec), 1);
                    //        int BatchLimit = 512;

                    //        Image ClusterImages = new Image(new int3(DimRec, DimRec, Math.Min(Indices.Length, BatchLimit)), true, true);
                    //        Image ClusterCTFs = new Image(new int3(DimRec, DimRec, Math.Min(Indices.Length, BatchLimit)), true);

                    //        for (int batchStart = 0; batchStart < Indices.Length; batchStart += BatchLimit)
                    //        {
                    //            int CurBatch = Math.Min(Indices.Length - batchStart, BatchLimit);

                    //            float[][] ImagesData = ClusterImages.GetHost(Intent.Write);
                    //            float[][] CTFsData = ClusterCTFs.GetHost(Intent.Write);
                    //            Matrix3[] CurRotations = new Matrix3[CurBatch];

                    //            for (int b = 0; b < CurBatch; b++)
                    //            {
                    //                int i = Indices[batchStart + b];
                    //                int p = i;

                    //                ImagesData[b] = RawParticles[p];
                    //                CTFsData[b] = RawCTFs[p];

                    //                CurRotations[b] = RawRotations[p];
                    //            }

                    //            foreach (var m in SymmetryMatrices)
                    //            {
                    //                float3[] Angles = Angles = CurRotations.Select(a => Matrix3.EulerFromMatrix(a * m)).ToArray();
                    //                Reconstructor.BackProject(ClusterImages, ClusterCTFs, Angles, new float3(1, 1, 0));
                    //            }
                    //        }

                    //        ClusterImages.Dispose();
                    //        ClusterCTFs.Dispose();

                    //        Image Rec = Reconstructor.Reconstruct(false, "C1");
                    //        Reconstructor.Dispose();

                    //        Rec.Multiply(Mask);

                    //        Reconstructions[ihalf] = Rec;
                    //    }

                    //    FSCs[isubset] = FSC.GetFSC(Reconstructions[0], Reconstructions[1]);

                    //    foreach (var rec in Reconstructions)
                    //        rec.Dispose();
                    //}

                    //Image FSCsImage = new Image(new[] { Helper.Combine(FSCs) }, new int3(FSCs[0].Length, FSCs.Length, 1));
                    //FSCsImage.WriteMRC(InputFolderPath + $"fsc_{suffix}.mrc", true);
                }
            }

            // Align one 3D map to another
            if (false)
            {
                Image MapTarget = Image.FromFile(@"E:\Dropbox\manuscripts\M\figures\denoising\emd_0339.mrc");
                MapTarget.MaskSpherically(180, 10, true);
                Image Mask = MapTarget.GetCopyGPU();
                Mask.Binarize(0.218f);
                MapTarget.Multiply(Mask);
                Mask.Dispose();

                Image MapDenoised = Image.FromFile(@"E:\Dropbox\manuscripts\M\figures\denoising\receptor_filtsharp.mrc");
                MapDenoised.MaskSpherically(180, 10, true);
                //Mask = MapDenoised.GetCopyGPU();
                //Mask.Binarize(0.000131f);
                //MapDenoised.Multiply(Mask);
                //Mask.Dispose();

                float3 Shift, Rotation;
                Image MapAligned = ImageHelper.AlignLocallyToTarget(MapDenoised, MapTarget, MapDenoised.Dims.X / 2 / 2 * 2, 2, out Shift, out Rotation);
                MapAligned.WriteMRC(@"E:\Dropbox\manuscripts\M\figures\denoising\receptor_filtsharp_aligned.mrc", MapDenoised.PixelSize, true);
            }

            // Test block-based Ewald sphere correction
            if (false)
            {
                Image Ref = Image.FromFile(@"F:\ewald\ref.mrc");
                Ref.MaskSpherically(130, 16, true);
                Projector RefProjector = new Projector(Ref, 1);

                //Ref.Symmetrize("O");
                //Ref.WriteMRC("d_ref_sym.mrc", true);

                int Dim = Ref.Dims.X;

                float3[] Angles = Helper.GetHealpixRotTilt(3, "O").Select(v => new float3(v.X, v.Y, 0)).ToArray();
                int NImages = Angles.Length;

                Random Rand = new Random(123);
                float[] Defoci = Helper.ArrayOfFunction(i => (float)Rand.NextDouble() * 0.2f + 0.4f, NImages);

                Image ProjectionsFT = new Image(new int3(Dim, Dim, NImages), true, true);
                Image CTFCoords = CTF.GetCTFCoords(Dim, Dim);
                Image CTFs = new Image(new int3(Dim, Dim, NImages), true);
                Image SliceCTFs = new Image(new int3(Dim, Dim, Dim), true);

                for (int i = 0; i < NImages; i++)
                {
                    Image RotatedFT = RefProjector.Project(new int3(Dim), new[] { Angles[i] * Helper.ToRad });
                    Image Rotated = RotatedFT.AsIFFT(true);
                    RotatedFT.Dispose();
                    Rotated.RemapFromFT(true);

                    Image SlicesFT = Rotated.AsFFT(false);
                    Rotated.Dispose();

                    CTF[] SliceParams = new CTF[Dim];
                    for (int s = 0; s < Dim; s++)
                        SliceParams[s] = new CTF()
                        {
                            PixelSize = 1M,
                            Defocus = (decimal)Defoci[i] - Math.Min(65, Math.Max(-65, (s - Dim / 2))) * 0.002M
                        };

                    GPU.CreateCTF(SliceCTFs.GetDevice(Intent.Write),
                                  CTFCoords.GetDevice(Intent.Read),
                                  IntPtr.Zero,
                                  (uint)CTFCoords.ElementsSliceComplex,
                                  SliceParams.Select(v => v.ToStruct()).ToArray(),
                                  false,
                                  (uint)Dim);
                    //SliceCTFs.Abs();
                    //SliceCTFs.WriteMRC("d_slicectfs.mrc", true);

                    SlicesFT.Multiply(SliceCTFs);

                    GPU.ReduceAdd(SlicesFT.GetDevice(Intent.Read),
                                  ProjectionsFT.GetDeviceSlice(i, Intent.Write),
                                  (uint)SlicesFT.ElementsSliceReal,
                                  (uint)Dim,
                                  1);
                    SlicesFT.Dispose();

                    GPU.CopyDeviceToDevice(SliceCTFs.GetDeviceSlice(Dim / 2, Intent.Read),
                                            CTFs.GetDeviceSlice(i, Intent.Write),
                                            SliceCTFs.ElementsSliceReal);
                }

                ProjectionsFT.Multiply(1f / (Dim * Dim));
                //Image Projections = ProjectionsFT.AsIFFT();
                //Projections.WriteMRC("d_projections.mrc", true);
                //CTFs.WriteMRC("d_ctfs.mrc", true);

                ProjectionsFT.ShiftSlices(Helper.ArrayOfConstant(new float3(Dim / 2, Dim / 2, 0), NImages));

                Matrix3[] SymMatrices = (new Symmetry("O")).GetRotationMatrices();

                float ObjectSize = 130 * 0.6f;
                int GridRes = 3;
                float GridStep = ObjectSize / Math.Max(1, (GridRes - 1));
                float GridOrigin = GridRes > 1 ? Dim / 2 - ObjectSize / 2 : Dim / 2;
                List<float3> RecCenters = new List<float3>();
                for (int z = 0; z < GridRes; z++)
                    for (int y = 0; y < GridRes; y++)
                        for (int x = 0; x < GridRes; x++)
                            RecCenters.Add(new float3(x, y, z) * GridStep + GridOrigin - Dim / 2);
                Image[] Reconstructions = new Image[RecCenters.Count];

                for (int icenter = 0; icenter < RecCenters.Count; icenter++)
                {
                    Projector Reconstructor = new Projector(new int3(Dim), 1);

                    foreach (var symMat in SymMatrices)
                    {
                        float3[] AnglesSym = Angles.Select(a => Matrix3.EulerFromMatrix(Matrix3.Euler(a * Helper.ToRad) * symMat) * Helper.ToDeg).ToArray();

                        CTF[] CTFParams = new CTF[NImages];
                        for (int i = 0; i < NImages; i++)
                        {
                            float3 Pos = RecCenters[icenter];
                            Matrix3 Rot = Matrix3.Euler(AnglesSym[i] * Helper.ToRad);
                            Pos = Rot * Pos;
                            CTFParams[i] = new CTF()
                            {
                                PixelSize = 1M,
                                Defocus = (decimal)Defoci[i] - (decimal)Pos.Z * 0.002M
                            };
                        }
                        GPU.CreateCTF(CTFs.GetDevice(Intent.Write),
                                      CTFCoords.GetDevice(Intent.Read),
                                      IntPtr.Zero,
                                      (uint)CTFCoords.ElementsSliceComplex,
                                      CTFParams.Select(v => v.ToStruct()).ToArray(),
                                      false,
                                      (uint)NImages);

                        Image ProjectionsCopy = ProjectionsFT.GetCopyGPU();

                        ProjectionsCopy.Multiply(CTFs);
                        CTFs.Multiply(CTFs);
                        Reconstructor.BackProject(ProjectionsCopy, CTFs, AnglesSym.Select(a => a * Helper.ToRad).ToArray(), new float3(1, 1, 0));

                        ProjectionsCopy.Dispose();
                        break;
                    }

                    Image LocalRec = Reconstructor.Reconstruct(false, "C1", -1, -1, -1, 0, true);
                    Reconstructor.Dispose();
                    LocalRec.MaskSpherically(130, 16, true);
                    //Rec.WriteMRC("d_rec_correction_XP_symmetrized.mrc", true);

                    LocalRec.FreeDevice();
                    Reconstructions[icenter] = LocalRec;
                    LocalRec.WriteMRC($"d_localrec_{icenter}.mrc", true);
                }

                Image Rec = new Image(new int3(Dim));
                float[][] RecData = Rec.GetHost(Intent.Write);
                float[][][] GridData = Reconstructions.Select(v => v.GetHost(Intent.Read)).ToArray();

                for (int z = 0; z < Dim; z++)
                {
                    float zz = (z - GridOrigin) / GridStep;
                    int Z0 = Math.Max(0, Math.Min(GridRes - 1, (int)Math.Floor(zz)));
                    int Z1 = Math.Max(0, Math.Min(GridRes - 1, Z0 + 1));

                    float InterpZ = Math.Max(0, Math.Min(1, zz - Z0));

                    for (int y = 0; y < Dim; y++)
                    {
                        float yy = (y - GridOrigin) / GridStep;
                        int Y0 = Math.Max(0, Math.Min(GridRes - 1, (int)Math.Floor(yy)));
                        int Y1 = Math.Max(0, Math.Min(GridRes - 1, Y0 + 1));

                        float InterpY = Math.Max(0, Math.Min(1, yy - Y0));

                        for (int x = 0; x < Dim; x++)
                        {
                            float xx = (x - GridOrigin) / GridStep;
                            int X0 = Math.Max(0, Math.Min(GridRes - 1, (int)Math.Floor(xx)));
                            int X1 = Math.Max(0, Math.Min(GridRes - 1, X0 + 1));

                            float InterpX = Math.Max(0, Math.Min(1, xx - X0));

                            int ElementXY = y * Dim + x;

                            float v000 = GridData[(Z0 * GridRes + Y0) * GridRes + X0][z][ElementXY];
                            float v001 = GridData[(Z0 * GridRes + Y0) * GridRes + X1][z][ElementXY];
                            float v010 = GridData[(Z0 * GridRes + Y1) * GridRes + X0][z][ElementXY];
                            float v011 = GridData[(Z0 * GridRes + Y1) * GridRes + X1][z][ElementXY];

                            float v100 = GridData[(Z1 * GridRes + Y0) * GridRes + X0][z][ElementXY];
                            float v101 = GridData[(Z1 * GridRes + Y0) * GridRes + X1][z][ElementXY];
                            float v110 = GridData[(Z1 * GridRes + Y1) * GridRes + X0][z][ElementXY];
                            float v111 = GridData[(Z1 * GridRes + Y1) * GridRes + X1][z][ElementXY];

                            float v00 = MathHelper.Lerp(v000, v001, InterpX);
                            float v01 = MathHelper.Lerp(v010, v011, InterpX);
                            float v10 = MathHelper.Lerp(v100, v101, InterpX);
                            float v11 = MathHelper.Lerp(v110, v111, InterpX);

                            float v0 = MathHelper.Lerp(v00, v01, InterpY);
                            float v1 = MathHelper.Lerp(v10, v11, InterpY);

                            float v = MathHelper.Lerp(v0, v1, InterpZ);

                            RecData[z][ElementXY] = v;
                        }
                    }
                }

                foreach (var rec in Reconstructions)
                    rec.Dispose();

                Rec.Symmetrize("O");

                Rec.WriteMRC("d_rec_correction_notsymmetrized.mrc", true);

                float[] F = FSC.GetFSC(Ref, Rec);
                (new Star(F, "wrpFSC")).Save("d_fsc_correction_notsymmetrized.star");
            }

            // Test Richard's Ewald sphere correction
            if (false)
            {
                CTF C = new CTF
                {
                    PixelSize = 0.6M,
                    Defocus = 0.5M,
                    DefocusDelta = 0.1M,
                    DefocusAngle = 30M
                };
                Image Coords = CTF.GetCTFCoords(434, 434);
                Image CoordsP = CTF.GetCTFPCoords(434, 434);

                float[] PQSigns = new float[Coords.ElementsComplex];
                float[] dummy = new float[Coords.ElementsComplex];

                C.GetEwaldLerpWeights(434, 0.6f).AsImaginary().WriteMRC("d_lerpweights.mrc", true);

                CTF.PrecomputePQSigns(434, 2, true, Coords.GetHostComplexCopy()[0], CoordsP.GetHostComplexCopy()[0], PQSigns);
                new Image(PQSigns, new int3(434, 434, 1), true).WriteMRC("d_pqsigns.mrc", true);

                Image ParticlesOri = Image.FromFile(@"D:\Dev\relion\particles.mrcs");
                ParticlesOri.ShiftSlices(Helper.ArrayOfConstant(new float3(434 / 2, 434 / 2, 0), ParticlesOri.Dims.Z));
                Image ParticlesFT = ParticlesOri.AsFFT();
                ParticlesFT.Multiply(1f / ParticlesOri.ElementsSliceReal);

                int PlanForw = GPU.CreateFFTPlan(ParticlesOri.Dims.Slice(), (uint)ParticlesOri.Dims.Z);
                int PlanBack = GPU.CreateIFFTPlan(ParticlesOri.Dims.Slice(), (uint)ParticlesOri.Dims.Z);

                Image Mask = new Image(ParticlesOri.Dims.Slice());
                Mask.Fill(1f);
                Mask.MaskSpherically(130 / (float)C.PixelSize, 3, false);
                //Mask.Multiply(-1);
                //Mask.Add(1);
                Mask.RemapToFT();
                Mask.WriteMRC("d_mask.mrc", true);

                Image CTFP = ParticlesFT.GetCopy();
                Image ParticlesFTCorr = ParticlesFT.GetCopy();
                Image ParticlesSuper = ParticlesOri.GetCopy();
                Image ParticlesCropped = ParticlesOri.GetCopy();
                Image MaskAngles = new Image(new int3(434, 434, 2), true);

                Image OutP = ParticlesFT.GetCopy();
                Image OutQ = ParticlesFT.GetCopy();

                CTF.ApplyPandQ(ParticlesFT,
                                Helper.ArrayOfConstant(C, ParticlesOri.Dims.Z),
                                ParticlesFTCorr,
                                ParticlesSuper,
                                ParticlesCropped,
                                CTFP,
                                MaskAngles,
                                Coords.GetHostComplexCopy()[0],
                                CoordsP.GetHostComplexCopy()[0],
                                Mask,
                                PlanForw,
                                PlanBack,
                                2,
                                false,
                                OutP,
                                OutQ);

                OutP.AsIFFT(false, 0, false, true).WriteMRC("d_outp.mrc", true);
                OutQ.AsIFFT(false, 0, false, true).WriteMRC("d_outq.mrc", true);

                GPU.CheckGPUExceptions();

                CTF.ApplyPandQPrecomp(ParticlesFT,
                                        Helper.ArrayOfConstant(C, ParticlesOri.Dims.Z),
                                        ParticlesFTCorr,
                                        ParticlesSuper,
                                        ParticlesCropped,
                                        CTFP,
                                        Coords,
                                        null,
                                        false,
                                        Mask,
                                        PlanForw,
                                        PlanBack,
                                        OutP,
                                        OutQ);

                //OutP.ShiftSlices(new[] { new float3(434 / 2, 434 / 2, 0) });
                //OutQ.ShiftSlices(new[] { new float3(434 / 2, 434 / 2, 0) });

                OutP.AsIFFT(false, 0, false, true).WriteMRC("d_outp_precomp.mrc", true);
                OutQ.AsIFFT(false, 0, false, true).WriteMRC("d_outq_precomp.mrc", true);

                //float[] CTFPData = C.Get2DP(Coords.GetHostComplexCopy()[0], CoordsP.GetHostComplexCopy()[0], true, 0);
                //CTFP = new Image(CTFPData, new int3(434, 434, 1), true, true);
                //CTFP.AsReal().WriteMRC("d_ctfp_real.mrc", true);
                //CTFP.AsImaginary().WriteMRC("d_ctfp_imag.mrc", true);

                float[] CTFWeightsData = new float[new int2(434).ElementsFFT()];
                C.GetEwaldWeights(Coords.GetHostComplexCopy()[0], 130, CTFWeightsData);
                Image CTFWeights = new Image(CTFWeightsData, new int3(434, 434, 1), true);
                CTFWeights.WriteMRC("d_ctf_ewaldweights.mrc", true);

                CTFWeights.Multiply(CTFWeights);

                GPU.CheckGPUExceptions();

                Projector Reconstructor = new Projector(new int3(434), 1);
                Reconstructor.BackProject(OutP, CTFWeights, new[] { new float3(0, 0, 0) }, new float3(1, 1, 0), C.GetEwaldRadius(434, 0.6f));
                Reconstructor.BackProject(OutQ, CTFWeights, new[] { new float3(0, 0, 0) }, new float3(1, 1, 0), -C.GetEwaldRadius(434, 0.6f));

                Reconstructor.Data.AsReal().WriteMRC("d_rec_real.mrc", true);
                Reconstructor.Data.AsImaginary().WriteMRC("d_rec_imag.mrc", true);
                Reconstructor.Weights.WriteMRC("d_rec_weights.mrc", true);


                Image Particle = new Image(new int3(512, 512, 1), true, true);
                Particle.Fill(new float2(1, 0));
                Image Weights = new Image(new int3(512, 512, 1), true);
                Weights.Fill(1f);


                //Projector Reconstructor = new Projector(new int3(512), 1);

                //Reconstructor.BackProject(Particle,
                //                          Weights,
                //                          new float3[] { new float3(0, 0, 0) },
                //                          new float3(1, 1, 0),
                //                          C.GetEwaldRadius(512));

                //Reconstructor.Data.WriteMRC("d_data.mrc", true);
                //Reconstructor.Weights.WriteMRC("d_weights.mrc", true);
            }

            // Test Zernike polynomials
            if (false)
            {
                //Image Result = new Image(new int3(512, 512, 9));
                //float[][] ResultData = Result.GetHost(Intent.Write);

                //for (int icoeff = 0; icoeff < Result.Dims.Z; icoeff++)
                //{
                //    int m, n;
                //    Zernike.EvenIndexToMN(icoeff, out m, out n);

                //    for (int y = 0; y < Result.Dims.Y; y++)
                //    {
                //        for (int x = 0; x < Result.Dims.X; x++)
                //        {
                //            float2 Coords = (new float2(x, y) / Result.Dims.X - 0.5f) * 2;

                //            ResultData[icoeff][y * Result.Dims.X + x] = (float)Zernike.Zcart(m, n, Coords.X, Coords.Y);
                //        }
                //    }
                //}

                //Console.WriteLine("");

                //for (int icoeff = 0; icoeff < Result.Dims.Z; icoeff++)
                //{
                //    int m, n;
                //    Zernike.EvenIndexToMN(icoeff, out m, out n);

                //    for (int y = 0; y < Result.Dims.Y; y++)
                //    {
                //        for (int x = 0; x < Result.Dims.X; x++)
                //        {
                //            float2 Coords = (new float2(x, y) / Result.Dims.X - 0.5f) * 2;

                //            ResultData[icoeff][y * Result.Dims.X + x] = (float)Zernike.Zcart(m, n, Coords.X, Coords.Y);
                //        }
                //    }
                //}

                //Result.MaskSpherically(Result.Dims.X - 6, 3, false);

                //Result.WriteMRC("d_zernike_even.mrc", true);

                //CTF C = new CTF()
                //{
                //    BeamTilt = new float2(1, 0)
                //};

                //Image BT = C.GetBeamTiltPhase(512, 512);
                ////BT.MaskSpherically(BT.Dims.X - 6, 3, false);
                //BT.WriteMRC("d_beamtilt.mrc", true);

                CTF F1 = new CTF()
                {
                    PixelSize = 0.6M,
                    Bfactor = -51M,
                    BfactorDelta = 51M,
                    BfactorAngle = -16M,
                    Cs = 0,
                    Amplitude = 1,
                    Defocus = 0
                };
                new Image(F1.Get2D(CTF.GetCTFCoordsFull(new int2(128), new int2(128)).GetHostComplexCopy()[0], false), new int3(128, 128, 1)).WriteMRC("d_ctf_frame1.mrc", true);

                CTF F2 = new CTF()
                {
                    PixelSize = 0.6M,
                    Bfactor = -7.5M,
                    BfactorDelta = 4M,
                    BfactorAngle = 21M,
                    Cs = 0,
                    Amplitude = 1,
                    Defocus = 0
                };
                new Image(F2.Get2D(CTF.GetCTFCoordsFull(new int2(128), new int2(128)).GetHostComplexCopy()[0], false), new int3(128, 128, 1)).WriteMRC("d_ctf_frame3.mrc", true);

                CTF C = new CTF();
                C.ZernikeCoeffsEven = new[]
                {
                    -0.9684984f,
                    0.01794727f,
                    -1.669379f,

                    0.05278453f,
                    -0.4586099f,
                    0.0008970127f,
                    -0.7982469f,
                    -0.3810967f
                };

                Image PC = C.GetGammaCorrection(0.6f, 512);
                PC.WriteMRC("d_gammacorrection.mrc", true);
                //new Image(PC.GetHostComplexCopy()[0].Select(c => (float)Math.Atan2(c.Y, c.X)).ToArray(), new int3(512, 512, 1), true).WriteMRC("d_phasecorrection.mrc", true);

                PC = C.GetPhaseCorrection(1, 512);
                PC = C.GetPhaseCorrection(1, 512);
                PC = C.GetPhaseCorrection(1, 512);
                PC = C.GetPhaseCorrection(1, 512);
            }

            // Test pseudoatom stuff
            if (false)
            {
                Image Ref = Image.FromFile(@"D:\Dev\pseudoatoms\ref.mrc");
                Ref.MaskSpherically(64, 10, true);
                Image Mask = Image.FromFile(@"D:\Dev\pseudoatoms\mask.mrc");

                Projector RefProjector = new Projector(Ref, 1);

                RandomNormal RandN = new RandomNormal(123);

                int Super = 4;

                float3[] Positions = PhysicsHelper.FillWithEquidistantPoints(Mask, (int)Mask.GetHostContinuousCopy().Sum() * 4, out _);
                int NAtoms = Positions.Length;

                float3[] Angles = Helper.GetHealpixRotTilt(4).Select(v => new float3(v.X, v.Y, 0) * Helper.ToRad).ToArray();

                Image RefProjections = RefProjector.ProjectToRealspace(new int2(Ref.Dims.X), Angles);
                //RefProjections.MaskSpherically(RefProjections.Dims.X / 2, 4, false);
                RefProjections.WriteMRC(@"D:\Dev\pseudoatoms\ref_projections.mrc", true);

                float DispSigma = 1;
                float3[][] Displacements = new float3[Angles.Length][];
                Helper.ForCPU(0, Angles.Length, 32, null, (a, threadID) =>
                {
                    RandomNormal ThreadRandN = new RandomNormal(a);
                    Displacements[a] = Helper.ArrayOfFunction(j => new float3(ThreadRandN.NextSingle(0, DispSigma),
                                                                              ThreadRandN.NextSingle(0, DispSigma),
                                                                              ThreadRandN.NextSingle(0, DispSigma)), NAtoms);
                }, null);

                Func<float3[], Image, float[]> GetIntensitiesFromVolume = (positions, volume) =>
                {
                    float[] Result = new float[NAtoms];

                    Image VolumeSuper = volume.AsScaled(volume.Dims * Super);
                    float[][] VolumeSuperData = VolumeSuper.GetHost(Intent.Read);

                    for (int p = 0; p < NAtoms; p++)
                    {
                        float3 Pos = positions[p] * Super;

                        int X0 = (int)Pos.X;
                        float ix = Pos.X - X0;
                        int X1 = X0 + 1;

                        int Y0 = (int)Pos.Y;
                        float iy = Pos.Y - Y0;
                        int Y1 = Y0 + 1;

                        int Z0 = (int)Pos.Z;
                        float iz = Pos.Z - Z0;
                        int Z1 = Z0 + 1;

                        float v000 = VolumeSuperData[Z0][Y0 * VolumeSuper.Dims.X + X0];
                        float v001 = VolumeSuperData[Z0][Y0 * VolumeSuper.Dims.X + X1];
                        float v010 = VolumeSuperData[Z0][Y1 * VolumeSuper.Dims.X + X0];
                        float v011 = VolumeSuperData[Z0][Y1 * VolumeSuper.Dims.X + X1];
                        float v100 = VolumeSuperData[Z1][Y0 * VolumeSuper.Dims.X + X0];
                        float v101 = VolumeSuperData[Z1][Y0 * VolumeSuper.Dims.X + X1];
                        float v110 = VolumeSuperData[Z1][Y1 * VolumeSuper.Dims.X + X0];
                        float v111 = VolumeSuperData[Z1][Y1 * VolumeSuper.Dims.X + X1];

                        float v00 = MathHelper.Lerp(v000, v001, ix);
                        float v01 = MathHelper.Lerp(v010, v011, ix);
                        float v10 = MathHelper.Lerp(v100, v101, ix);
                        float v11 = MathHelper.Lerp(v110, v111, ix);

                        float v0 = MathHelper.Lerp(v00, v01, iy);
                        float v1 = MathHelper.Lerp(v10, v11, iy);

                        float v = MathHelper.Lerp(v0, v1, iz);

                        Result[p] = v;
                    }

                    VolumeSuper.Dispose();

                    return Result;
                };

                Func<float3[], float[], Image> RasterizeToVolume = (positions, intensities) =>
                {
                    Image VolumeSuper = new Image(Ref.Dims * Super);
                    float[][] VolumeData = VolumeSuper.GetHost(Intent.ReadWrite);

                    Image WeightsSuper = new Image(VolumeSuper.Dims);
                    float[][] WeightsData = WeightsSuper.GetHost(Intent.ReadWrite);

                    for (int p = 0; p < positions.Length; p++)
                    {
                        float3 Pos = positions[p] * Super;

                        int X0 = (int)Pos.X;
                        float ix = Pos.X - X0;
                        int X1 = X0 + 1;

                        int Y0 = (int)Pos.Y;
                        float iy = Pos.Y - Y0;
                        int Y1 = Y0 + 1;

                        int Z0 = (int)Pos.Z;
                        float iz = Pos.Z - Z0;
                        int Z1 = Z0 + 1;

                        float v0 = 1.0f - iz;
                        float v1 = iz;

                        float v00 = (1.0f - iy) * v0;
                        float v10 = iy * v0;
                        float v01 = (1.0f - iy) * v1;
                        float v11 = iy * v1;

                        float v000 = (1.0f - ix) * v00;
                        float v100 = ix * v00;
                        float v010 = (1.0f - ix) * v10;
                        float v110 = ix * v10;
                        float v001 = (1.0f - ix) * v01;
                        float v101 = ix * v01;
                        float v011 = (1.0f - ix) * v11;
                        float v111 = ix * v11;

                        VolumeData[Z0][Y0 * VolumeSuper.Dims.X + X0] += intensities[p] * v000;
                        VolumeData[Z0][Y0 * VolumeSuper.Dims.X + X1] += intensities[p] * v001;
                        VolumeData[Z0][Y1 * VolumeSuper.Dims.X + X0] += intensities[p] * v010;
                        VolumeData[Z0][Y1 * VolumeSuper.Dims.X + X1] += intensities[p] * v011;

                        VolumeData[Z1][Y0 * VolumeSuper.Dims.X + X0] += intensities[p] * v100;
                        VolumeData[Z1][Y0 * VolumeSuper.Dims.X + X1] += intensities[p] * v101;
                        VolumeData[Z1][Y1 * VolumeSuper.Dims.X + X0] += intensities[p] * v110;
                        VolumeData[Z1][Y1 * VolumeSuper.Dims.X + X1] += intensities[p] * v111;

                        WeightsData[Z0][Y0 * VolumeSuper.Dims.X + X0] += v000;
                        WeightsData[Z0][Y0 * VolumeSuper.Dims.X + X1] += v001;
                        WeightsData[Z0][Y1 * VolumeSuper.Dims.X + X0] += v010;
                        WeightsData[Z0][Y1 * VolumeSuper.Dims.X + X1] += v011;

                        WeightsData[Z1][Y0 * VolumeSuper.Dims.X + X0] += v100;
                        WeightsData[Z1][Y0 * VolumeSuper.Dims.X + X1] += v101;
                        WeightsData[Z1][Y1 * VolumeSuper.Dims.X + X0] += v110;
                        WeightsData[Z1][Y1 * VolumeSuper.Dims.X + X1] += v111;
                    }

                    for (int z = 0; z < VolumeData.Length; z++)
                        for (int i = 0; i < VolumeData[z].Length; i++)
                            VolumeData[z][i] /= Math.Max(1e-10f, WeightsData[z][i]);

                    WeightsSuper.Dispose();

                    Image Volume = VolumeSuper.AsScaled(Ref.Dims);
                    VolumeSuper.Dispose();

                    return Volume;
                };

                float[] Intensities = GetIntensitiesFromVolume(Positions, Ref);

                // Test deterioration from repeated rasterization
                if (false)
                    for (int ideteriorate = 0; ideteriorate < 5; ideteriorate++)
                    {
                        Image Rasterized = RasterizeToVolume(Positions, Intensities);
                        Rasterized.WriteMRC($@"D:\Dev\pseudoatoms\rasterized_iter{ideteriorate:D2}.mrc", true);

                        Intensities = GetIntensitiesFromVolume(Positions, Rasterized);

                        Rasterized.MaskSpherically(64, 10, true);

                        float[] Corr = FSC.GetFSC(Ref, Rasterized);
                        new Star(Corr, "fsc").Save($@"D:\Dev\pseudoatoms\fsc_rasterized_vs_ref_iter{ideteriorate:D2}.star");
                    }

                Func<float3[], float[], float3[][], Image> ProjectAtoms = (positions, intensities, displacements) =>
                {
                    Image AtomProjectionsSuper = new Image(new int3(Ref.Dims.X * Super, Ref.Dims.Y * Super, Angles.Length));
                    float[][] AtomProjectionsData = AtomProjectionsSuper.GetHost(Intent.ReadWrite);

                    Parallel.For(0, Angles.Length, a =>
                    {
                        Matrix3 R = Matrix3.Euler(Angles[a]);
                        float3 CenterOri = new float3(Ref.Dims / 2);
                        float3 CenterSuper = CenterOri * Super;

                        float[] ProjectionData = AtomProjectionsData[a];

                        for (int p = 0; p < NAtoms; p++)
                        {
                            float3 Displacement = new float3(0, 0, 0);
                            if (displacements != null)
                                Displacement = displacements[a][p];

                            float3 Transformed = R * (positions[p] + Displacement - CenterOri) * Super + CenterSuper;

                            int X0 = (int)Transformed.X;
                            float ix = Transformed.X - X0;
                            int X1 = X0 + 1;

                            int Y0 = (int)Transformed.Y;
                            float iy = Transformed.Y - Y0;
                            int Y1 = Y0 + 1;

                            float v0 = 1.0f - iy;
                            float v1 = iy;

                            float v00 = (1.0f - ix) * v0;
                            float v10 = ix * v0;
                            float v01 = (1.0f - ix) * v1;
                            float v11 = ix * v1;

                            ProjectionData[Y0 * AtomProjectionsSuper.Dims.X + X0] += intensities[p] * v00;
                            ProjectionData[Y0 * AtomProjectionsSuper.Dims.X + X1] += intensities[p] * v01;
                            ProjectionData[Y1 * AtomProjectionsSuper.Dims.X + X0] += intensities[p] * v10;
                            ProjectionData[Y1 * AtomProjectionsSuper.Dims.X + X1] += intensities[p] * v11;
                        }
                    });

                    Image Result = AtomProjectionsSuper.AsScaled(new int2(Ref.Dims.X));
                    AtomProjectionsSuper.Dispose();

                    return Result;
                };

                // Calculate FRC between atoms projections and reference
                if (true)
                {
                    Image AtomProjections = ProjectAtoms(Positions, Intensities, Displacements);
                    AtomProjections.MaskSpherically(AtomProjections.Dims.X / 2, 4, false);
                    AtomProjections.WriteMRC(@"D:\Dev\pseudoatoms\atom_projections.mrc", true);

                    Image AtomProjectionsFT = AtomProjections.AsFFT();
                    float[][] AtomProjectionsFTData = AtomProjectionsFT.GetHost(Intent.Read);
                    Image RefProjectionsFT = RefProjections.AsFFT();
                    float[][] RefProjectionsFTData = RefProjectionsFT.GetHost(Intent.Read);

                    float3[] Shells = new float3[AtomProjections.Dims.X / 2];
                    for (int a = 0; a < Angles.Length; a++)
                    {
                        float[] AData = AtomProjectionsFTData[a];
                        float[] RData = RefProjectionsFTData[a];

                        int i = 0;
                        Helper.ForEachElementFT(new int2(AtomProjections.Dims.X), (x, y, xx, yy, r, angle) =>
                        {
                            int R = (int)Math.Round(r);
                            if (R >= Shells.Length)
                                return;

                            float2 A = new float2(AData[i * 2], AData[i * 2 + 1]);
                            float2 B = new float2(RData[i * 2], RData[i * 2 + 1]);

                            float AB = A.X * B.X + A.Y * B.Y;
                            float A2 = A.LengthSq();
                            float B2 = B.LengthSq();

                            Shells[R] += new float3(AB, A2, B2);

                            i++;
                        });
                    }

                    float[] FRC = Shells.Select(v => v.X / (float)Math.Max(1e-16, Math.Sqrt(v.Y * v.Z))).ToArray();
                    new Star(FRC, "frc").Save(@"D:\Dev\pseudoatoms\frc_atoms4x_vs_ref_masked.star");
                }

                // Compare reference to Fourier-space reconstruction from its projections
                if (false)
                {
                    Image RefProjectionsFT = RefProjections.AsFFT();
                    RefProjectionsFT.ShiftSlices(Helper.ArrayOfConstant(new float3(Ref.Dims.X / 2, Ref.Dims.Y / 2, 0), Angles.Length));

                    Image Weights = new Image(RefProjectionsFT.Dims, true);
                    Weights.Fill(1);

                    Projector Reconstructor = new Projector(Ref.Dims, 1);
                    Reconstructor.BackProject(RefProjectionsFT, Weights, Angles, new float3(1, 1, 0));

                    RefProjectionsFT.Dispose();
                    Weights.Dispose();

                    Image Rec = Reconstructor.Reconstruct(false);
                    Reconstructor.Dispose();

                    Rec.MaskSpherically(64, 10, true);
                    Rec.WriteMRC(@"D:\Dev\pseudoatoms\rec_fourier.mrc", true);

                    float[] Corr = FSC.GetFSC(Rec, Ref);
                    Rec.Dispose();
                    new Star(Corr, "fsc").Save(@"D:\Dev\pseudoatoms\fsc_rec_vs_ref.star");
                }

                RefProjections = ProjectAtoms(Positions, Intensities, Displacements);

                // Pre-weight reference projections by a very naive sampling function that assumes perfectly even angular distribution
                if (true)
                {
                    float[] Weights = new float[Ref.Dims.X / 2];

                    for (int r = 0; r < Weights.Length; r++)
                    {
                        float Sum = 0;

                        float3 Point = new float3(r, 0, 0);

                        for (int a = 0; a < Angles.Length; a++)
                        {
                            float3 Normal = Matrix3.Euler(Angles[a]).Transposed() * float3.UnitZ;
                            float Dist = Math.Abs(float3.Dot(Normal, Point));
                            Sum += Math.Max(0, 1 - Dist);
                        }

                        Weights[r] = 1f / Math.Max(1, Sum);
                    }

                    Image Weights2D = new Image(new int3(Ref.Dims.X, Ref.Dims.Y, 1), true);
                    float[] Weights2DData = Weights2D.GetHost(Intent.ReadWrite)[0];
                    int i = 0;
                    Helper.ForEachElementFT(new int2(Ref.Dims.X), (x, y, xx, yy, r, angle) =>
                    {
                        Weights2DData[i++] = Weights[Math.Min(Weights.Length - 1, (int)Math.Round(r))];
                    });

                    Image RefProjectionsFT = RefProjections.AsFFT();
                    RefProjectionsFT.MultiplySlices(Weights2D);

                    RefProjections = RefProjectionsFT.AsIFFT(false, 0, true);
                    RefProjections.WriteMRC(@"D:\Dev\pseudoatoms\ref_projections_weighted.mrc", true);
                }

                Intensities = Helper.ArrayOfFunction(i => RandN.NextSingle(0, 0f), NAtoms);

                for (int isirt = 0; isirt < 20; isirt++)
                {
                    Image Predictions = ProjectAtoms(Positions, Intensities, Displacements);
                    Predictions.WriteMRC($@"D:\Dev\pseudoatoms\sirt_disp_pred_iter{isirt:D2}.mrc", true);
                    Predictions.Subtract(RefProjections);
                    Predictions.Multiply(-0.1f / Angles.Length);
                    Predictions.WriteMRC($@"D:\Dev\pseudoatoms\sirt_disp_diff_iter{isirt:D2}.mrc", true);

                    Image DiffSuper = Predictions.AsScaled(new int2(Ref.Dims.X * Super));
                    Predictions.Dispose();

                    float[][] DiffData = DiffSuper.GetHost(Intent.Read);

                    int NThreads = 32;
                    float[][] ThreadUpdates = Helper.ArrayOfFunction(i => new float[NAtoms], NThreads);

                    Helper.ForCPU(0, Angles.Length, NThreads, null, (a, threadID) =>
                    {
                        float[] Diff = DiffData[a];

                        Matrix3 R = Matrix3.Euler(Angles[a]);
                        float3 CenterOri = new float3(Ref.Dims / 2);
                        float3 CenterSuper = CenterOri * Super;

                        for (int p = 0; p < NAtoms; p++)
                        {
                            float3 Transformed = R * (Positions[p] + Displacements[a][p] - CenterOri) * Super + CenterSuper;

                            int X0 = (int)Transformed.X;
                            float ix = Transformed.X - X0;
                            int X1 = X0 + 1;

                            int Y0 = (int)Transformed.Y;
                            float iy = Transformed.Y - Y0;
                            int Y1 = Y0 + 1;

                            float v00 = Diff[Y0 * DiffSuper.Dims.X + X0];
                            float v01 = Diff[Y0 * DiffSuper.Dims.X + X1];
                            float v10 = Diff[Y1 * DiffSuper.Dims.X + X0];
                            float v11 = Diff[Y1 * DiffSuper.Dims.X + X1];

                            float v0 = MathHelper.Lerp(v00, v01, ix);
                            float v1 = MathHelper.Lerp(v10, v11, ix);

                            float v = MathHelper.Lerp(v0, v1, iy);

                            ThreadUpdates[threadID][p] += v;
                        }
                    }, null);

                    foreach (var update in ThreadUpdates)
                        for (int i = 0; i < update.Length; i++)
                            Intensities[i] += update[i];

                    DiffSuper.Dispose();

                    Image Reconstruction = RasterizeToVolume(Positions, Intensities);
                    Reconstruction.MaskSpherically(64, 10, true);
                    Reconstruction.WriteMRC($@"D:\Dev\pseudoatoms\sirt_disp_rec_iter{isirt:D2}.mrc", true);

                    float[] Corr = FSC.GetFSC(Reconstruction, Ref);
                    new Star(Corr, "fsc").Save($@"D:\Dev\pseudoatoms\fsc_sirt_disp_vs_ref_iter{isirt:D3}.star");

                    Reconstruction.Dispose();
                }
            }

            // Test EER
            if (false)
            {
                int2 Dims = new int2(8192);
                int NFrames = 40;
                int NThreads = 2;
                float[][] Result = Helper.ArrayOfFunction(i => new float[Dims.Elements()], NFrames);

                HeaderTiff H = new HeaderTiff(@"F:\FoilHole_24015405_Data_24016401_24016403_20200225_0014_Fractions.mrc.eer");

                Stopwatch W = new Stopwatch();
                W.Start();

                Helper.ForCPU(0, NFrames, NThreads, null, (i, threadID) =>
                {
                    EERNative.ReadEER(@"F:\FoilHole_24015405_Data_24016401_24016403_20200225_0014_Fractions.mrc.eer", i * 10, (i + 1) * 10, 2, Result[i]);
                }, null);

                W.Stop();
                Console.WriteLine(W.Elapsed.TotalSeconds);

                Image EER = new Image(Result, new int3(Dims.X, Dims.Y, 40));
                EER.WriteMRC("d_eer.mrc", true);
            }

            // Make MDOC for Flo's EER data
            if (false)
            {
                string MdocText = File.ReadAllText(@"E:\eer_tomo\mdoc\Position_87_header.mdoc");
                string Template = File.ReadAllText(@"E:\eer_tomo\mdoc\template.txt");

                int Z = 0;

                foreach (var path in Directory.EnumerateFiles(@"E:\eer_tomo\mnt\Krios3\Tomo_FS_batch1", "*.eer"))
                {
                    FileInfo Info = new FileInfo(path);
                    DateTime Timestamp = Info.LastWriteTime;

                    string Name = Helper.PathToName(path);
                    string DATETIME = Timestamp.ToString("yy-MMM-dd  HH:mm:ss");
                    string ZVAL = (Z++).ToString();
                    string ANGLE = Name.Substring(Name.IndexOf('[') + 1, Name.IndexOf(']') - Name.IndexOf('[') - 1);

                    string Modified = Template.Replace("$ZVAL", ZVAL).Replace("$DATETIME", DATETIME).Replace("$ANGLE", ANGLE).Replace("$FILENAME", Helper.PathToNameWithExtension(path));
                    MdocText += "\n\n";
                    MdocText += Modified;
                }

                File.WriteAllText(@"E:\eer_tomo\mdoc\Position_87.mdoc", MdocText);
            }

            // Figure out defect correction
            if (false)
            {
                Image Gain = Image.FromFile(@"E:\mrc4bit\gain\CountingGainRef.mrc");
                Image Defects = Image.FromFile(@"E:\mrc4bit\gain\defects.mrc");
                Image Corrected = Gain.GetCopyGPU();

                DefectModel DM = new DefectModel(Defects, 3);
                DM.Correct(Gain, Corrected);

                Corrected.WriteMRC(@"E:\mrc4bit\gain\corrected.mrc", true);
            }

            // Compare EwaldWeights CPU and GPU implementations
            if (false)
            {
                Image Coords = CTF.GetCTFCoords(512, 512);
                CTF C = new CTF();

                Image ResultCPU = new Image(Coords.Dims, true);
                ResultCPU.GetHost(Intent.Write)[0] = C.GetEwaldWeights(Coords.GetHostComplexCopy()[0], 200);
                ResultCPU.WriteMRC("d_ewaldcpu.mrc", true);

                Image ResultGPU = new Image(Coords.Dims, true);
                GPU.CreateCTFEwaldWeights(ResultGPU.GetDevice(Intent.Write),
                    Coords.GetDevice(Intent.Read),
                    IntPtr.Zero,
                    200,
                    (uint)Coords.ElementsSliceComplex,
                    new[] { C.ToStruct() },
                    1);
                ResultGPU.WriteMRC("d_ewaldgpu.mrc", true);
            }

            // Florian 😣
            if (false)
            {
                Image Coords = CTF.GetCTFCoords(512, 512);
                CTF C1 = new CTF();
                CTF C2 = new CTF() { Defocus = C1.Defocus + 0.5M };
                CTF C3 = new CTF() { Defocus = C1.Defocus + 1 };
                Image Ref = new Image(new int3(512, 512, 3), true);
                GPU.CreateCTF(Ref.GetDevice(Intent.Write),
                                Coords.GetDevice(Intent.Read),
                                IntPtr.Zero,
                                (uint)Coords.ElementsSliceComplex,
                                new[] { C1.ToStruct(), C2.ToStruct(), C3.ToStruct() },
                                false,
                                3);
                Ref.Multiply(Ref);
                Ref.WriteMRC("d_ref.mrc", true);

                Image Weights = Ref.AsReducedAlongZ();
                Weights.Multiply(3);
                Weights.Max(0.02f);
                Weights.WriteMRC("d_weights.mrc", true);

                RandomNormal Rand = new RandomNormal(123);
                Image Fit = new Image(new int3(512, 512, 1), true);
                Fit.TransformValues(v => Rand.NextSingle(0, 1));
                Fit.WriteMRC("d_fit_init.mrc", true);

                for (int i = 0; i < 100; i++)
                {
                    Image Diff = Ref.GetCopy();
                    Diff.MultiplySlices(Fit);

                    Diff.Subtract(Ref);
                    Diff.Multiply(-0.2f);
                    Image DiffSum = Diff.AsReducedAlongZ();
                    DiffSum.Multiply(3);

                    DiffSum.Divide(Weights);

                    Fit.Add(DiffSum);
                    if ((i + 1) % 10 == 0)
                        Fit.WriteMRC($"d_fit_{i:D3}.mrc", true);

                    Diff.Dispose();
                    DiffSum.Dispose();
                }
            }

            // Test bimodal distribution fitting for correlation score histograms
            if (false)
            {
                Image CorrImage = Image.FromFile(@"F:\liang_ribos\TS_028_15.00Apx_70Sref_corr.mrc");
                float[] CorrOri = CorrImage.GetHostContinuousCopy();
                float[] CorrNoZero = CorrOri.Where(v => v != 0).ToArray();
                float Min = MathHelper.Min(CorrNoZero);
                float Max = MathHelper.Max(CorrNoZero);
                float[] Histogram = MathHelper.Histogram(CorrNoZero, 100).Select(v => (float)Math.Log(Math.Max(1, v))).ToArray();
                float[] FakeDist = Helper.Combine(Helper.ArrayOfSequence(0, Histogram.Length, 1).Select(i => Helper.ArrayOfConstant(MathHelper.Lerp(Min, Max, (float)i / (Histogram.Length - 1)), (int)(Histogram[i] * 100))).ToArray());

                //double[][] Corr = CorrImage.GetHostContinuousCopy().Where(v => v != 0).Select(v => new double[] { v }).ToArray();

                GaussianMixtureModel GMM = new GaussianMixtureModel(2);
                var Clusters = GMM.Learn(FakeDist.Select(v => new double[] { v }).ToArray());

                int[] Classes = Clusters.Decide(CorrNoZero.Select(v => new double[] { v }).ToArray());
                //float[] Probs = Clusters.(Corr).Select(a => (float)a[0]).ToArray();

                int[] ClassHist = MathHelper.Histogram(Classes.Select(v => (float)v), 2);

                int[] Indices = Helper.ArrayOfSequence(0, CorrOri.Length, 1).Where(i => CorrOri[i] != 0).ToArray();
                for (int i = 0; i < Indices.Length; i++)
                    CorrOri[Indices[i]] = Classes[i];

                //float Threshold = (float)GMM.Gaussians.Means[1][0];
                //for (int i = 0; i < CorrOri.Length; i++)
                //{
                //    CorrOri[i] = CorrOri[i] > Threshold ? 1 : 0;
                //}

                CorrImage = new Image(CorrOri, CorrImage.Dims);
                CorrImage.WriteMRC(@"F:\liang_ribos\classes.mrc", true);
            }

            // Calculate aliasing-free CTF box sizes for Bob's book
            if (false)
            {
                List<List<float>> Results = new List<List<float>>();

                for (decimal defocus = 0.3M; defocus <= 4; defocus += 0.3M)
                {
                    List<float> Result = new List<float>();

                    for (decimal res = 6; res >= 1; res -= 0.1M)
                    {
                        CTF C = new CTF()
                        {
                            Voltage = 200,
                            Defocus = (decimal)defocus,
                            Cs = 2.7M
                        };

                        Result.Add((float)Math.Round(C.GetAliasingFreeSize((float)res) * 0.75f / 2) * 2);
                    }

                    Results.Add(Result);
                }

                Star TableOut = new Star(Results.Select(a => a.Select(v => v.ToString()).ToArray()).ToArray(), Helper.ArrayOfFunction(i => $"column{i}", Results.Count));
                TableOut.Save("d_minbox.star");
            }

            // Do k-means on Felix' multi-body results
            if (false)
            {
                Star TableIn = new Star(@"E:\felix_multibody\run_ct13_data.star");

                List<string> PCANameColumn = new List<string>();
                List<List<float>> PCAColumns = new List<List<float>>();
                using (var Reader = File.OpenText(@"E:\felix_multibody\analyse_projections_along_eigenvectors_all_particles.txt"))
                {
                    string Line;
                    while ((Line = Reader.ReadLine()) != null)
                    {
                        string[] Parts = Line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                        if (Parts.Length < 1)
                            continue;

                        PCANameColumn.Add(Parts[0]);
                        for (int i = 1; i < Parts.Length; i++)
                        {
                            if (PCAColumns.Count < i)
                                PCAColumns.Add(new List<float>());
                            PCAColumns[i - 1].Add(float.Parse(Parts[i], CultureInfo.InvariantCulture));
                        }
                    }
                }
                int NComp = PCAColumns.Count;
                int NParts = PCANameColumn.Count;
                int NClasses = 20;

                int[] UseComps = { 0, 1 };

                Accord.Math.Random.Generator.Seed = 0;

                double[][] Observations = Helper.ArrayOfFunction(i => UseComps.Select(v => (double)PCAColumns[v][i]).ToArray(), NParts);

                KMeans ClusterAlg = new KMeans(NClasses);
                var Clusters = ClusterAlg.Learn(Observations);
                int[] Memberships = Clusters.Decide(Observations);

                List<int> ClusterMapping = Helper.ArrayOfSequence(0, NClasses, 1).ToList();
                ClusterMapping.Sort((a, b) =>
                {
                    int Result = 0;
                    double[] CentroidA = Clusters.Centroids[a];
                    double[] CentroidB = Clusters.Centroids[b];
                    for (int i = 0; i < CentroidA.Length; i++)
                    {
                        int Comp = CentroidA[i].CompareTo(CentroidB[i]);
                        if (Comp != 0)
                        {
                            Result = Comp;
                            break;
                        }
                    }
                    return Result;
                });

                double[][] SortedCentroids = ClusterMapping.Select(i => Clusters.Centroids[i]).ToArray();
                int[][] SortedClasses = ClusterMapping.Select(c => Helper.ArrayOfSequence(0, NParts, 1).Where(p => Memberships[p] == c).ToArray()).ToArray();

                using (var WriterClassInfo = File.CreateText(@"E:\felix_multibody\MBClasses\classes.txt"))
                using (var WriterScript = File.CreateText(@"E:\felix_multibody\MBClasses\reconstruct.txt"))
                {
                    WriterScript.Write("trap \"exit\" INT\n");
                    WriterScript.Write("mkdir MBClasses/halfmaps\n");

                    for (int i = 0; i < NClasses; i++)
                    {
                        Star TableOut = TableIn.CreateSubset(SortedClasses[i]);
                        TableOut.Save($@"E:\felix_multibody\MBClasses\class{i:D3}.star");

                        WriterScript.Write($"mpirun -n 16 relion_reconstruct_mpi --i MBClasses/class{i:D3}.star --o MBClasses/halfmaps/run_half1_class{i:D3}_unfil.mrc --ctf --subset 1 --skip_gridding --maxres 5\n");
                        WriterScript.Write($"mpirun -n 16 relion_reconstruct_mpi --i MBClasses/class{i:D3}.star --o MBClasses/halfmaps/run_half2_class{i:D3}_unfil.mrc --ctf --subset 2 --skip_gridding --maxres 5\n");
                        WriterScript.Write($"mkdir MBClasses/postprocess{i:D3}\n");
                        WriterScript.Write($"relion_postprocess --mask MBClasses/mask.mrc --i MBClasses/halfmaps/run_half1_class{i:D3}_unfil.mrc --o MBClasses/postprocess{i:D3}/postprocess{i:D3} --angpix 1.05 --skip_fsc_weighting --low_pass -1\n");
                        WriterScript.Write($"cp MBClasses/postprocess{i:D3}/postprocess{i:D3}.mrc MBClasses/\n");
                        WriterScript.Write($"relion_image_handler --i MBClasses/postprocess{i:D3}.mrc --o MBClasses/postprocess{i:D3}.mrc --rescale_angpix 2\n");

                        WriterClassInfo.WriteLine(i + ":\t" + string.Join("\t", SortedCentroids[i].Select(v => v.ToString("F3", CultureInfo.InvariantCulture))) + "\t" + SortedClasses[i].Length);
                    }
                }
            }

            // Test PyTorch stuff
            if (false)
            {
                Torch.SetSeed(1);
                Console.WriteLine(Torch.IsCudaAvailable());
                Console.WriteLine(Torch.IsCudnnAvailable());
                Console.WriteLine(Torch.TryInitializeDeviceType(DeviceType.CUDA));

                //using (var Blob = Float32Tensor.Ones(new long[] { 8, 1, 64, 64, 64}, DeviceType.CUDA, 0))
                //{
                //    TorchTensor[] Scattered = Blob.ScatterToDevices(0, new long[] { 0, 1, 2, 3 });
                //}

                Func<Sequential> MakeBla = () =>
                {
                    Sequential Bla = Sequential();
                    Bla.Add("C1", TorchSharp.NN.Modules.Conv2D(1, 1024, 11, 1, 5));
                    Bla.Add("C2", TorchSharp.NN.Modules.Conv2D(1024, 1024, 11, 1, 5));
                    Bla.Add("C3", TorchSharp.NN.Modules.Conv2D(1024, 1024, 11, 1, 5));
                    Bla.Add("C4", TorchSharp.NN.Modules.Conv2D(1024, 1024, 11, 1, 5));
                    Bla.Add("C5", TorchSharp.NN.Modules.Conv2D(1024, 1024, 11, 1, 5));
                    Bla.Add("C6", TorchSharp.NN.Modules.Conv2D(1024, 1024, 11, 1, 5));

                    return Bla;
                };

                Console.WriteLine(GPU.GetFreeMemory(0));
                UNet2D Model0 = UNet2D(1, 1, 1, 1);
                Console.WriteLine(GPU.GetFreeMemory(0));
                Torch.CudaEmptyCache();
                Console.WriteLine(GPU.GetFreeMemory(0));
                Model0.ToCuda(0);
                Console.WriteLine(GPU.GetFreeMemory(0));
                Model0.Dispose();
                Console.WriteLine(GPU.GetFreeMemory(0));
                Model0 = UNet2D(1, 1, 1, 1);
                Model0.ToCuda(0);
                Model0.Dispose();
                Console.WriteLine(GPU.GetFreeMemory(0));
                Model0 = UNet2D(1, 1, 1, 1);
                Model0.ToCuda(0);
                Model0.Dispose();
                Console.WriteLine(GPU.GetFreeMemory(0));
                Torch.CudaEmptyCache();
                Console.WriteLine(GPU.GetFreeMemory(0));

                //Console.WriteLine(GPU.GetFreeMemory(0));
                //Sequential Model0 = MakeBla();
                //Console.WriteLine(GPU.GetFreeMemory(0));
                //Model0.ToCuda(0);
                //Console.WriteLine(GPU.GetFreeMemory(0));
                //Model0.Dispose();
                //Console.WriteLine(GPU.GetFreeMemory(0));
                //Model0 = MakeBla();
                //Model0.ToCuda(0);
                //Model0.Dispose();
                //Console.WriteLine(GPU.GetFreeMemory(0));
                //Model0 = MakeBla();
                //Model0.ToCuda(0);
                //Model0.Dispose();
                //Console.WriteLine(GPU.GetFreeMemory(0));
                //Torch.CudaEmptyCache();
                //Console.WriteLine(GPU.GetFreeMemory(0));

                TorchTensor FakeInput = Float32Tensor.Random(new long[] { 2, 1, 512, 512 }, DeviceType.CUDA, 0);
                TorchTensor Result = Model0.Forward(FakeInput);
                Console.WriteLine(GPU.GetFreeMemory(0));

                Result.Dispose();
                Result = Model0.Forward(FakeInput);
                Console.WriteLine(GPU.GetFreeMemory(0));
                Result.Dispose();
                FakeInput.Dispose();
                Model0.Dispose();
                Console.WriteLine(GPU.GetFreeMemory(0));

                Torch.CudaEmptyCache();

                Console.WriteLine(GPU.GetFreeMemory(0));


                using (var train = TorchSharp.Data.Loader.MNIST(_dataLocation, _trainBatchSize))
                using (var test = TorchSharp.Data.Loader.MNIST(_dataLocation, _testBatchSize, false))
                using (var model = new Model("model"))
                using (var model2 = new Model("model"))
                using (var optimizer = TorchSharp.NN.Optimizer.Adam(model.GetParameters(), 0.01, 1e-4))
                {
                    model.ToCuda(0);

                    TorchTensor Fake = Float32Tensor.RandomN(new long[] { 3, 1, 64, 64, 64 }, DeviceType.CUDA);

                    TorchTensor FakeResult = model.Forward(Fake);

                    //model.Save("savetest.pt");
                    //model.Load("savetest.pt");

                    optimizer.SetLearningRateAdam(0.1);

                    Stopwatch sw = new Stopwatch();
                    sw.Start();

                    for (var epoch = 1; epoch <= _epochs; epoch++)
                    {
                        Train(model, optimizer, NLL(), train, epoch, _trainBatchSize, train.Size());
                        sw.Stop();
                        Console.WriteLine($"Elapsed time {sw.ElapsedMilliseconds}.");

                        Test(model, NLL(reduction: TorchSharp.NN.Reduction.Sum), test, test.Size());

                        sw.Restart();
                    }

                    Console.ReadLine();
                }
            }

            // Fake micrographs from Ying's data
            if (false)
            {
                int3 DimsMicFull = new int3(4096, 4096, 1);
                int3 DimsMicScaled = new int3(3358, 3358, 1);
                int DimParticle = 256;
                int DimCTF = 3358;

                float AngPixOri = 1.23f;
                float AngPixScaled = 1.5f;

                int NClasses = 6;

                HashSet<string> UniqueMicNames;
                {
                    Star TableIn = new Star(@"F:\noiseGAN\c4_coords.star", "particles");
                    string[] MicNames = TableIn.GetColumn("rlnMicrographName").Select(v => Helper.PathToName(v)).ToArray();
                    UniqueMicNames = Helper.GetUniqueElements(MicNames);
                }

                Projector[] Refs = new Projector[NClasses];
                for (int i = 0; i < NClasses; i++)
                {
                    Image Ref = Image.FromFile($@"F:\noiseGAN\maps\c{i}.mrc");
                    Ref.MaskSpherically(140, 60, true);
                    Refs[i] = new Projector(Ref, 1);
                    Refs[i].FreeDevice();
                    Ref.Dispose();
                }

                int BatchSize = 8;

                Image CompositeFT = new Image(DimsMicScaled, true, true);
                Image CTFCoords = CTF.GetCTFCoords(DimCTF, DimCTF);
                Image CTFSim = new Image(new int3(DimCTF, DimCTF, BatchSize), true);
                Image ProjFT = new Image(new int3(DimParticle, DimParticle, BatchSize), true, true);
                Image Proj = new Image(new int3(DimParticle, DimParticle, BatchSize));
                Image ProjPadded = new Image(new int3(DimCTF, DimCTF, BatchSize));
                Image ProjPaddedFT = new Image(new int3(DimCTF, DimCTF, BatchSize), true, true);

                int PlanForw = GPU.CreateFFTPlan(new int3(DimParticle, DimParticle, 1), (uint)BatchSize);
                int PlanBack = GPU.CreateIFFTPlan(new int3(DimParticle, DimParticle, 1), (uint)BatchSize);
                int PlanForwPadded = GPU.CreateFFTPlan(new int3(DimCTF, DimCTF, 1), (uint)BatchSize);

                string[][] AllMicNames = new string[NClasses][];
                float2[][] AllPositions = new float2[NClasses][];
                float3[][] AllAngles = new float3[NClasses][];
                CTF[][] AllCTFs = new CTF[NClasses][];

                for (int c = 0; c < NClasses; c++)
                {
                    Star TableIn = new Star(@$"F:\noiseGAN\c{c}_coords.star", "particles");

                    string[] MicNames = TableIn.GetColumn("rlnMicrographName").Select(v => Helper.PathToName(v)).ToArray();
                    AllMicNames[c] = MicNames;

                    float2[] Coords = TableIn.GetRelionCoordinates().Select(v => new float2(v.X, v.Y)).ToArray();
                    float2[] Offsets = TableIn.GetRelionOffsets().Select(v => new float2(v.X, v.Y)).ToArray();
                    for (int i = 0; i < Coords.Length; i++)
                        Coords[i] -= Offsets[i];

                    AllPositions[c] = Coords;

                    float3[] Angles = TableIn.GetRelionAngles().Select(v => v * Helper.ToRad).ToArray();
                    AllAngles[c] = Angles;

                    CTF[] CTFs = TableIn.GetRelionCTF();
                    for (int i = 0; i < CTFs.Length; i++)
                        CTFs[i].PixelSize = (decimal)AngPixScaled;

                    AllCTFs[c] = CTFs;
                }

                foreach (var mic in UniqueMicNames)
                {
                    CompositeFT.Fill(new float2(0, 0));

                    for (int c = 0; c < NClasses; c++)
                    {
                        string[] MicNames = AllMicNames[c];
                        int[] Rows = Helper.ArrayOfSequence(0, MicNames.Length, 1).Where(v => MicNames[v] == mic).ToArray();

                        float2[] Positions = Helper.IndexedSubset(AllPositions[c], Rows);
                        float3[] Angles = Helper.IndexedSubset(AllAngles[c], Rows);
                        CTF[] CTFs = Helper.IndexedSubset(AllCTFs[c], Rows);

                        int NParticles = Rows.Length;

                        for (int b = 0; b < NParticles; b += BatchSize)
                        {
                            int CurBatch = Math.Min(BatchSize, NParticles - b);

                            GPU.CreateCTF(CTFSim.GetDevice(Intent.Write),
                                        CTFCoords.GetDevice(Intent.Read),
                                        IntPtr.Zero,
                                        (uint)CTFSim.ElementsSliceComplex,
                                        CTFs.Skip(b).Take(CurBatch).Select(v => v.ToStruct()).ToArray(),
                                        false,
                                        (uint)CurBatch);

                            Refs[c].Project(new int2(DimParticle), Angles.Skip(b).Take(CurBatch).ToArray(), ProjFT);

                            GPU.IFFT(ProjFT.GetDevice(Intent.Read),
                                    Proj.GetDevice(Intent.Write),
                                    Proj.Dims.Slice(),
                                    (uint)BatchSize,
                                    PlanBack,
                                    false);

                            GPU.SphereMask(Proj.GetDevice(Intent.Read),
                                            Proj.GetDevice(Intent.Write),
                                            Proj.Dims.Slice(),
                                            80,
                                            40,
                                            true,
                                            (uint)BatchSize);

                            GPU.PadFTFull(Proj.GetDevice(Intent.Read),
                                            ProjPadded.GetDevice(Intent.Write),
                                            Proj.Dims.Slice(),
                                            ProjPadded.Dims.Slice(),
                                            (uint)CurBatch);

                            GPU.FFT(ProjPadded.GetDevice(Intent.Read),
                                    ProjPaddedFT.GetDevice(Intent.Write),
                                    ProjPadded.Dims.Slice(),
                                    (uint)BatchSize,
                                    PlanForwPadded);

                            GPU.ShiftStackFT(ProjPaddedFT.GetDevice(Intent.Read),
                                             ProjPaddedFT.GetDevice(Intent.Write),
                                             ProjPadded.Dims.Slice(),
                                             Helper.ToInterleaved(Positions.Skip(b).Take(CurBatch).Select(v => new float3(v.X, v.Y, 0)).ToArray()),
                                             (uint)CurBatch);

                            ProjPaddedFT.Multiply(CTFSim);

                            for (int i = 0; i < CurBatch; i++)
                                GPU.AddToSlices(CompositeFT.GetDevice(Intent.Read),
                                                ProjPaddedFT.GetDeviceSlice(i, Intent.Read),
                                                CompositeFT.GetDevice(Intent.Write),
                                                (uint)CompositeFT.ElementsSliceReal,
                                                1);
                        }

                        Refs[c].FreeDevice();
                    }

                    Image Composite = CompositeFT.AsPadded(new int2(DimsMicFull), true).AsIFFT(false, 0, true).AndDisposeParent();
                    Composite.Multiply(-1);
                    Composite.WriteMRC(Path.Combine(@"F:\noiseGAN\simraw", mic + ".mrc"), true);
                    Composite.Dispose();
                }
            }

            // Make MDOCs for FEI Tomo files  g11001_001[-0.00]-941853.tif
            if (false)
            {
                string Folder = @"X:\dtegunov\corona_insitu\1907session";
                string[] FileNames = Directory.EnumerateFiles(Folder, "*.tif").Select(s => Helper.PathToNameWithExtension(s)).ToArray();
                var UniqueRoots = Helper.GetUniqueElements(FileNames.Select(s => s.Substring(0, s.IndexOf("_"))));

                foreach (var root in UniqueRoots)
                {
                    string[] Names = FileNames.Where(s => s.Contains(root)).ToArray();
                    int NTilts = Names.Length;
                    float[] Angles = Names.Select(s =>
                    {
                        int Start = s.IndexOf("[") + 1;
                        int End = s.IndexOf("]");
                        return float.Parse(s.Substring(Start, End - Start));
                    }).ToArray();
                    int[] Numbers = Names.Select(s =>
                    {
                        int Start = s.IndexOf("]") + 2;
                        int End = s.IndexOf(".tif");
                        return int.Parse(s.Substring(Start, End - Start));
                    }).ToArray();

                    List<int> IndicesSortedDose = Helper.ArrayOfSequence(0, NTilts, 1).ToList();
                    IndicesSortedDose.Sort((a, b) => Numbers[a].CompareTo(Numbers[b]));

                    using (TextWriter Writer = File.CreateText(Path.Combine(Folder, root + ".mdoc")))
                    {
                        Writer.WriteLine("PixelSpacing = 1.7005\n" +
                                         "Voltage = 300\n" +
                                         "ImageFile = TS_01.mrc\n" +
                                         "ImageSize = 3708 3838\n" +
                                         "DataMode = 1\n" +
                                         "\n" +
                                         "[T = SerialEM: Digitized on EMBL Krios 2                     09 - Nov - 18  21:43:34]\n" +
                                         "\n" +
                                         "[T = Tilt axis angle = 8, binning = 1  spot = 8  camera = 1]\n");

                        for (int i = 0; i < NTilts; i++)
                        {
                            int SortedIndex = IndicesSortedDose[i];

                            DateTime StartTime = new DateTime(2020, 12, 12, 9, 0, 0) + new TimeSpan(0, 0, i);

                            Writer.WriteLine($"[ZValue = {i}]\n" +
                                             $"TiltAngle = {Angles[SortedIndex]}\n" +
                                             $"ExposureDose = {i * 2.2}\n" +
                                             "PixelSpacing = 3.51\n" +
                                             "ExposureTime = 1.2\n" +
                                             $"SubFramePath = {Names[SortedIndex]}\n" +
                                             $"DateTime = {StartTime.ToString("dd-MMM-yy  HH:mm:ss")}\n\n");
                        }
                    }
                }
            }

            // WTF FFT???
            if (false)
            {
                //Image DummyFT = new Image(new int3(5760, 4092, 1), true, true);
                //DummyFT.Fill(new float2(1, 0));
                //DummyFT.WriteMRC("d_dummyftbefore.mrc", true);
                //Image Dummy = DummyFT.AsIFFT(false, 100);
                //DummyFT.WriteMRC("d_dummyftafter.mrc", true);


                Movie M = new Movie(@"E:\wtffft\a2020-09-26_15.52.14_HH713_grid8_48-5_0001_30.0.tif");
                Image Stack = Image.FromFile(@"E:\wtffft\a2020-09-26_15.52.14_HH713_grid8_48-5_0001_30.0.tif");

                M.ExportMovie(Stack, new ProcessingOptionsMovieExport() { PixelSizeX = 0.834M, PixelSizeY = 0.834M, DoDenoise = true, DoDenoiseDeconv = true, DoAverage = true });
            }

            // Simulate ribos in Liang's tomograms for Nat Meth cover
            if (false)
            {
                Species S = Species.FromFile(@"Z:\em\dtegunov\julia_ribos\population\species\9ba9bb3d-3c68-4fd9-874a-4173b07e3e49\70S.species");

                Image Ref = S.MapDenoised.AsScaled(new int3(new float3(S.MapDenoised.Dims) * (float)S.PixelSize / 10 + 1) / 2 * 2).AndFreeParent();
                Projector Proj = new Projector(Ref, 2, 3);

                foreach (var tomoName in Helper.GetUniqueElements(S.Particles.Select(p => p.SourceName)))
                {
                    int3 DimsTomo = new int3(632, 632, 272);
                    Image VolumeTomo = new Image(DimsTomo);

                    Particle[] ParticlesTomo = S.Particles.Where(p => p.SourceName == tomoName).ToArray();

                    foreach (var particle in ParticlesTomo)
                    {
                        int3 Position = new int3((int)Math.Round(particle.Coordinates[0].X / 10),
                                                 (int)Math.Round(particle.Coordinates[0].Y / 10),
                                                 (int)Math.Round(particle.Coordinates[0].Z / 10));

                        Image Rotated = Proj.Project(Ref.Dims, new[] { Matrix3.EulerFromMatrix(Matrix3.Euler(particle.Angles[0]).Transposed()) }).AsIFFT(true).AndDisposeParent();
                        Rotated.RemapFromFT(true);

                        //VolumeTomo.Insert(Rotated, Position);
                        Rotated.Dispose();
                    }

                    VolumeTomo.WriteMRC(@"Z:\em\dtegunov\julia_ribos\simulated\" + tomoName + ".mrc", 10, true);
                    VolumeTomo.Dispose();
                }
            }

            // Test 2D rotation and shift
            if (false)
            {
                Image Ref = new Image(new int3(64, 64, 1));
                Ref.GetHost(Intent.ReadWrite)[0][32 * 64 + 32] = 1;

                Image Trans = new Image(Ref.Dims);
                Image Rot = new Image(Ref.Dims);

                GPU.ShiftAndRotate2D(Ref.GetDevice(Intent.Read),
                                     Trans.GetDevice(Intent.Write),
                                     new int2(Ref.Dims),
                                     new float[] { 4, 4 },
                                     new float[] { 0 },
                                     1);

                Image Ref3D = new Image(Helper.ArrayOfConstant(Trans.GetHost(Intent.Read)[0], 64), new int3(64));
                Projector Projector = new Projector(Ref3D, 2);
                Image Proj2D = Projector.ProjectToRealspace(new int2(64), new float3[] { new float3(0, 0, 90) * Helper.ToRad });

                GPU.ShiftAndRotate2D(Trans.GetDevice(Intent.Read),
                                     Rot.GetDevice(Intent.Write),
                                     new int2(Ref.Dims),
                                     new float[] { -4, -4 },
                                     new float[] { 90 * Helper.ToRad },
                                     1);

                Trans.WriteMRC("d_trans.mrc", true);
                Rot.WriteMRC("d_rot.mrc", true);
                Proj2D.WriteMRC("d_proj2d.mrc", true);
            }

            // Test peak finding
            if (false)
            {
                Image Proto = new Image(new int3(64, 64, 2));
                Proto.GetHost(Intent.ReadWrite)[0][32 * 64 + 32] = 1;
                Proto.GetHost(Intent.ReadWrite)[1][32 * 64 + 32] = 1;
                //Proto.ShiftSlices(new[] { new float3(10, 10, 0), new float3(0) });

                Image Replica = new Image(new int3(64, 64, 64));
                GPU.Repeat(Proto.GetDevice(Intent.Read),
                            Replica.GetDevice(Intent.Write),
                            (uint)Proto.ElementsSliceReal,
                            32,
                            2);

                Replica.WriteMRC("d_replica.mrc", true);

                Random Rand = new Random(123);
                float3[] Shifts = Helper.ArrayOfFunction(i => new float3((float)Rand.NextDouble() * 10 - 5, (float)Rand.NextDouble() * 10 - 5, 0), 64);
                Replica.ShiftSlices(Shifts);

                IntPtr Positions = GPU.MallocDevice(Replica.Dims.Z * 3);
                GPU.ValueFill(Positions, Replica.Dims.Z * 3, 99);
                IntPtr Values = GPU.MallocDevice(Replica.Dims.Z);

                GPU.PeakOne2D(Replica.GetDevice(Intent.Read),
                    Positions,
                    Values,
                    new int2(64),
                    new int2(16),
                    true,
                    Replica.Dims.Z);

                float[] h_Positions = new float[Replica.Dims.Z * 3];
                GPU.CopyDeviceToHost(Positions, h_Positions, h_Positions.Length);

                List<float> Error = new List<float>();
                for (int i = 0; i < 64; i++)
                {
                    Error.Add(Math.Abs(h_Positions[i * 3 + 0] - Shifts[i].X));
                    Error.Add(Math.Abs(h_Positions[i * 3 + 1] - Shifts[i].Y));
                }
            }

            // Test C2DNet alignment
            if (false)
            {
                GPU.SetDevice(1);

                IntPtr NamePtr = GPU.GetDeviceName(1);
                string Name = Marshal.PtrToStringAnsi(NamePtr);
                CPU.HostFree(NamePtr);

                Image RefVol = Image.FromFile(@"E:\particleWGAN\c4.mrc").AsScaled(new int3(64)).AndDisposeParent();
                RefVol.MaskSpherically(32, 8, true);
                Projector Proj = new Projector(RefVol, 2);

                Random Rand = new Random(123);
                RandomNormal RandN = new RandomNormal(234);
                int Batch = 64;

                float3[] Angles = Helper.ArrayOfFunction(i => new float3(0, 0, (float)Rand.NextDouble() * 360), Batch);
                float3[] Shifts = Helper.ArrayOfFunction(i => new float3(RandN.NextSingle(0, 2), RandN.NextSingle(0, 2), 0), Batch);

                Image RefFT = Proj.Project(new int2(64), Angles.Select(v => v * Helper.ToRad).ToArray());
                GPU.DeviceSynchronize();
                RefFT.WriteMRC("d_ref_before.mrc", true);
                Image RefIFT = RefFT.AsIFFT(false, 0, false, true);
                GPU.DeviceSynchronize();
                RefIFT.RemapFromFT();
                RefIFT.WriteMRC("d_ref.mrc", true);
                RefFT.WriteMRC("d_ref_after.mrc", true);

                Image DataFT = Proj.Project(new int2(64), Helper.ArrayOfFunction(i => new float3(0, 0, 0) * Helper.ToRad, Batch));
                DataFT.ShiftSlices(Shifts);
                Image DataFTCopy = DataFT.GetCopyGPU();
                Image DataIFT = DataFTCopy.AsIFFT(false, 0, false, true);
                DataIFT.RemapFromFT();
                DataIFT.WriteMRC("d_data.mrc", true);

                float minshell = 8;
                float shiftstep = 32 / 8 * 3;
                float anglestep = (float)Math.Asin(1f / minshell) * 3 * Helper.ToDeg;
                int anglesteps = (int)Math.Ceiling(360 / anglestep);
                anglestep = 360f / anglesteps;

                List<float3> InitPoses = new List<float3>();
                for (int b = 0; b < Batch; b++)
                    for (int angle = 0; angle < anglesteps; angle++)
                        for (int y = -1; y <= 1; y++)
                            for (int x = -1; x <= 1; x++)
                                InitPoses.Add(new float3(x * shiftstep, y * shiftstep, angle * anglestep));
                IntPtr d_InitPoses = GPU.MallocDeviceFromHost(Helper.ToInterleaved(InitPoses.ToArray()), InitPoses.Count * 3);

                Image DataAligned = new Image(IntPtr.Zero, new int3(64, 64, Batch));

                GPU.C2DNetAlign(RefFT.GetDevice(Intent.Read),
                                RefFT.Dims.X,
                                1,
                                DataFT.GetDevice(Intent.Read),
                                DataIFT.GetDevice(Intent.Read),
                                IntPtr.Zero,
                                DataFT.Dims.X,
                                d_InitPoses,
                                InitPoses.Count / Batch,
                                8,
                                32,
                                6,
                                anglestep / 3,
                                shiftstep / 3,
                                16,
                                Batch,
                                DataAligned.GetDevice(Intent.Write),
                                IntPtr.Zero);

                float[] h_FinalPoses = new float[Batch * 3];
                GPU.CopyDeviceToHost(d_InitPoses, h_FinalPoses, h_FinalPoses.Length);
                float3[] FinalPoses = Helper.FromInterleaved3(h_FinalPoses);

                float MaxErrAngle = 0;
                float MaxErrShift = 0;
                for (int b = 0; b < Batch; b++)
                {
                    MaxErrShift = Math.Max(Math.Abs(FinalPoses[b].X + Shifts[b].X), MaxErrShift);
                    MaxErrShift = Math.Max(Math.Abs(FinalPoses[b].Y + Shifts[b].Y), MaxErrShift);
                    MaxErrAngle = Math.Max(Math.Abs((FinalPoses[b].Z + 360) % 360 - Angles[b].Z), MaxErrAngle);
                }

                DataAligned.WriteMRC("d_dataaligned.mrc", true);
            }

            // Test FEI MRC header bug
            if (true)
            {
                Image Input = Image.FromFile(@"E:\pranav_debug\Position_40_001[13.00]_Fractions.mrc", 0);

                Input.WriteMRC(@"E:\pranav_debug\out.mrc", true);
            }
        }

        private class Model : CustomModule
        {
            private UNet3D conv1 = UNet3D(1, 1, 1);

            public Model(string name) : base(name)
            {
                RegisterModule("conv1", conv1);
            }

            public override TorchTensor Forward(TorchTensor input)
            {
                using (var l11 = conv1.Forward(input))
                {
                    return LogSoftMax(l11, 1);
                }
            }
        }

        private static void Train(
            Model model,
            TorchSharp.NN.Optimizer optimizer,
            Loss loss,
            IEnumerable<(TorchTensor, TorchTensor)> dataLoader,
            int epoch,
            long batchSize,
            long size)
        {
            model.Train();

            int batchId = 1;

            foreach (var (data, target) in dataLoader)
            {
                Stopwatch sw = new Stopwatch();
                sw.Start();

                using (var d_data = data.Cuda())
                using (var d_target = target.Cuda())
                {
                    //for (int i = 0; i < 118; i++)
                    {
                        optimizer.ZeroGrad();

                        TorchTensor FakeData = Float32Tensor.Random(new long[] { 64, 1, 256, 256 }, DeviceType.CUDA);

                        using (var prediction = model.Forward(FakeData))
                        using (var output = loss(prediction, d_target))
                        {
                            output.Backward();

                            optimizer.Step();

                            if (batchId % _logInterval == 0)
                            {
                                Console.WriteLine($"\rTrain: epoch {epoch} [{batchId * batchSize} / {size}] Loss: {output.ToSingle()}");
                            }

                            batchId++;
                        }
                    }
                }

                sw.Stop();
                //Console.WriteLine($"Elapsed time {sw.ElapsedMilliseconds}.");

                data.Dispose();
                target.Dispose();
            }
        }

        private static void Test(
                                Model model,
                                Loss loss,
                                IEnumerable<(TorchTensor, TorchTensor)> dataLoader,
                                long size)
        {
            model.Eval();

            double testLoss = 0;
            int correct = 0;

            foreach (var (data, target) in dataLoader)
            {
                using (var prediction = model.Forward(data))
                using (var output = loss(prediction, target))
                {
                    testLoss += output.ToSingle();

                    var pred = prediction.Argmax(1);

                    correct += pred.Eq(target).Sum().ToInt32(); // Memory leak here

                    data.Dispose();
                    target.Dispose();
                    pred.Dispose();
                }

            }

            Console.WriteLine($"\rTest set: Average loss {testLoss / size} | Accuracy {(double)correct / size}");
        }
    }
}
