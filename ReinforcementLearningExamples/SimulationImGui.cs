using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ImGuiNET;
using MachineLearning;
using Microsoft.Xna.Framework;

namespace Arrows;

public static class SimulationImGui
{
    private const int MaxRewardGraphSize = 100;

    private static bool showRewardGraphs;
    private static bool showNeuralNetwork;

    private static List<float> rewardBests = [];
    private static List<float> rewardAverages = [];
    private static List<float> rewardMedians = [];

    private static bool bestArrowSelected;
    public static Arrow SelectedArrow;

    public static void DrawImGui(Simulation simulation, GameTime gameTime)
    {
        ImGui.Begin("Simulation");

        ImGui.SeparatorText("Settings");

        ImGui.Text($"Arrow count: {simulation.Arrows.Length}");

        ImGuiUtils.ComboEnum("Network hidden layers activation function", ref simulation.HiddenLayersActivationFunction);
        ImGuiUtils.ComboEnum("Network output layers activation function", ref simulation.OutputLayerActivationFunction);

        ImGui.SliderFloat("Q-Learner gain", ref simulation.QLearnerGain, 0f, 1f);

        if (ImGui.DragFloat("Time between resets", ref simulation.NewTimeBetweenResets, 0.1f, 1f))
            simulation.TimeLeftBeforeReset = MathF.Min(simulation.TimeLeftBeforeReset, simulation.NewTimeBetweenResets);
        if (ImGui.Checkbox("Uncap FPS", ref simulation.SimulationSpeedUncapped))
        {
            Application.Instance.Graphics.SynchronizeWithVerticalRetrace = !simulation.SimulationSpeedUncapped;
            Application.Instance.Graphics.ApplyChanges();
        }

        if (!simulation.SimulationSpeedUncapped)
            ImGui.BeginDisabled();

        ImGui.SliderFloat("Simulation FPS", ref simulation.SimulationFrameRate, 5f, 300f);

        if (!simulation.SimulationSpeedUncapped)
            ImGui.EndDisabled();

        ImGui.SeparatorText("Readonly data");

        double fps = 1.0 / gameTime.ElapsedGameTime.TotalSeconds;
        ImGui.Text($"FPS: {fps}");
        ImGui.Text($"Total time: {gameTime.TotalGameTime}");
        ImGui.Text($"Current iteration: {simulation.CurrentIteration}");
        ImGui.Text($"Best reward: {(rewardBests.Count > 0 ? rewardBests[^1] : 0f)}");
        ImGui.Text($"Average reward: {(rewardAverages.Count > 0 ? rewardAverages[^1] : 0f)}");
        ImGui.Text($"Median reward: {(rewardMedians.Count > 0 ? rewardMedians[^1] : 0f)}");
        double simulationSpeed = simulation.SimulationSpeedUncapped ? fps / simulation.SimulationFrameRate : 1f;
        ImGui.Text($"Running at {simulationSpeed.ToString("F2", CultureInfo.CurrentCulture)}x speed");
        ImGui.Text($"{(simulationSpeed / simulation.TimeBetweenResets).ToString("F3", CultureInfo.CurrentCulture)} iterations per second");

        ImGui.TextColored(Color.Orange.ToVector4().ToNumerics(), $"Next reset in {simulation.TimeLeftBeforeReset}s");
        if (ImGui.Button("Randomize arrow spawn"))
            simulation.RandomizeArrowSpawn();
        ImGui.Text($"Target position {simulation.TargetPosition}");
        ImGui.Text($"Spawn position {simulation.StartingArrowPosition}");

        ImGui.SeparatorText("Actions");

        ImGui.Checkbox("Running", ref simulation.Running);

        if (ImGui.Button("Next generation"))
            simulation.ResetSimulation(true);

        if (simulation.Running)
            ImGui.BeginDisabled();

        if (ImGui.Button("Next frame"))
            simulation.RunningForOneFrame = true;

        if (simulation.Running)
            ImGui.EndDisabled();

        if (ImGui.Button("Save best"))
            simulation.SaveNetwork();

        if (ImGui.Button("Load save"))
            simulation.LoadSavedNetwork();

        ImGui.Checkbox("Show reward graphs", ref showRewardGraphs);

        if (ImGui.Button("Select best arrow"))
            SelectedArrow = simulation.Arrows[0];

        if (ImGui.Checkbox("Keep best arrow selected", ref bestArrowSelected) && !bestArrowSelected)
            SelectedArrow = null;

        if (bestArrowSelected)
            SelectedArrow = simulation.Arrows[0];

        if (SelectedArrow != null)
        {
            ImGui.SeparatorText("Selected arrow data");

            ImGui.Text($"Position {SelectedArrow.Position}");
            ImGui.Text($"Angle {MathHelper.ToDegrees(SelectedArrow.Angle)}deg");
            Vector2 direction = new(MathF.Cos(SelectedArrow.Angle), -MathF.Sin(SelectedArrow.Angle));
            ImGuiUtils.DirectionVector("Direction", ref direction, SelectedArrow.TargetDirection);

            if (ImGui.Button("+"))
                SelectedArrow = simulation.Arrows[(simulation.Arrows.IndexOf(SelectedArrow) - 1 + simulation.Arrows.Length) % simulation.Arrows.Length];
            ImGui.SameLine();
            if (ImGui.Button("-"))
                SelectedArrow = simulation.Arrows[(simulation.Arrows.IndexOf(SelectedArrow) + 1) % simulation.Arrows.Length];
            ImGui.SameLine();
            ImGui.Text($"Reward {SelectedArrow.TotalReward}");

            ImGui.Checkbox("Show neural network", ref showNeuralNetwork);

            if (showNeuralNetwork)
                DrawNeuralNetworkWindow();
        }

        ImGui.End();

        if (showRewardGraphs)
            DrawFitnessGraphWindow();
    }

    private static void DrawNeuralNetworkWindow()
    {
        ImGui.Begin("Neural Network");
        ImGuiUtils.DisplayNeuralNetwork(SelectedArrow.Network);
        ImGui.End();
    }

    private static void DrawFitnessGraphWindow()
    {
        ImGui.Begin("Reward graphs");

        ref float pointer = ref Unsafe.NullRef<float>();
        if (rewardBests.Count > 0)
            pointer = ref CollectionsMarshal.AsSpan(rewardBests)[0];

        ImGui.PlotLines(
            "Best",
            ref pointer,
            rewardBests.Count,
            0,
            string.Empty,
            float.MaxValue,
            float.MaxValue,
            new(ImGui.GetContentRegionAvail().X - 50f, 50f)
        );

        if (rewardAverages.Count > 0)
            pointer = ref CollectionsMarshal.AsSpan(rewardAverages)[0];

        ImGui.PlotLines(
            "Average",
            ref pointer,
            rewardAverages.Count,
            0,
            string.Empty,
            float.MaxValue,
            float.MaxValue,
            new(ImGui.GetContentRegionAvail().X - 50f, 50f)
        );

        if (rewardMedians.Count > 0)
            pointer = ref CollectionsMarshal.AsSpan(rewardMedians)[0];

        ImGui.PlotLines(
            "Median",
            ref pointer,
            rewardMedians.Count,
            0,
            string.Empty,
            float.MaxValue,
            float.MaxValue,
            new(ImGui.GetContentRegionAvail().X - 50f, 50f)
        );

        ImGui.End();
    }

    public static void UpdateRewardGraphsData(Simulation simulation)
    {
        rewardBests.Add((float) simulation.Arrows.First().TotalReward);
        rewardAverages.Add((float) simulation.Arrows.Average(n => n.TotalReward));
        rewardMedians.Add((float) simulation.Arrows[simulation.Arrows.Length / 2].TotalReward);

        if (rewardBests.Count > MaxRewardGraphSize)
            rewardBests = rewardBests[^MaxRewardGraphSize..];

        if (rewardAverages.Count > MaxRewardGraphSize)
            rewardAverages = rewardAverages[^MaxRewardGraphSize..];

        if (rewardMedians.Count > MaxRewardGraphSize)
            rewardMedians = rewardMedians[^MaxRewardGraphSize..];
    }
}
