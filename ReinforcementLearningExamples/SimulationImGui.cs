using System;
using System.Collections.Generic;
using System.Globalization;
using ImGuiNET;
using MachineLearning;
using MachineLearning.NeuralNetwork;
using Microsoft.Xna.Framework;

namespace Arrows;

public static class SimulationImGui
{
    private const int MaxRewardGraphSize = 100;

    private static bool showRewardGraphs;
    private static bool showNeuralNetwork;
    private static bool showQLearnerNeuralNetwork;

    private static List<float> rewardMedians = [];
    private static List<float> estimatedRewardMedians = [];

    public static void DrawImGui(Simulation simulation, GameTime gameTime)
    {
        ImGui.Begin("Simulation");

        ImGui.SeparatorText("Settings");

        ImGui.Text($"Arrow count: {simulation.Arrows.Length}");
        ImGui.Checkbox("Draw all arrows", ref simulation.DrawAllArrows);

        ImGui.SliderFloat("Q-Learner gain", ref simulation.QLearnerGain, 0f, 1f);

        if (ImGui.DragFloat("Time between resets", ref simulation.NewTimeBetweenResets, 0.1f, 1f))
            simulation.TimeLeftBeforeReset = MathF.Min(simulation.TimeLeftBeforeReset, simulation.NewTimeBetweenResets);
        if (ImGui.Checkbox("Uncap FPS", ref simulation.SimulationSpeedUncapped))
        {
            Application.Instance.TargetElapsedTime = TimeSpan.FromSeconds(1.0 / simulation.SimulationFrameRate);
            Application.Instance.IsFixedTimeStep = simulation.SimulationSpeedUncapped;
            Application.Instance.Graphics.SynchronizeWithVerticalRetrace = !simulation.SimulationSpeedUncapped;
            Application.Instance.Graphics.ApplyChanges();
        }

        if (!simulation.SimulationSpeedUncapped)
            ImGui.BeginDisabled();

        if (ImGui.SliderFloat("Simulation FPS", ref simulation.SimulationFrameRate, 60f, 3000f))
            Application.Instance.TargetElapsedTime = TimeSpan.FromSeconds(1.0 / simulation.SimulationFrameRate);

        if (!simulation.SimulationSpeedUncapped)
            ImGui.EndDisabled();

        ImGui.SeparatorText("Readonly data");

        if (simulation.UpdatingQLearner)
            ImGui.TextDisabled("Updating Q-Learner...");
        else
            ImGui.Spacing();

        double fps = 1.0 / gameTime.ElapsedGameTime.TotalSeconds;
        ImGui.Text($"FPS: {fps}");
        ImGui.Text($"Total time: {gameTime.TotalGameTime}");
        ImGui.Text($"Current iteration: {simulation.CurrentIteration}");
        double simulationSpeed = simulation.SimulationSpeedUncapped ? fps / 60.0 : 1.0;
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

        ImGui.Checkbox("Show Q-Learner neural network", ref showQLearnerNeuralNetwork);

        ImGui.SeparatorText("MainArrow data");

        Arrow mainArrow = simulation.MainArrow;
        ImGui.Text($"Position {mainArrow.Position}");
        ImGui.Text($"Angle {MathHelper.ToDegrees(mainArrow.Angle % MathHelper.TwoPi)}deg");
        Vector2 direction = new(MathF.Cos(mainArrow.Angle), -MathF.Sin(mainArrow.Angle));
        ImGuiUtils.DirectionVector("Direction", ref direction, mainArrow.TargetDirection);

        ImGui.Text($"Total reward {mainArrow.TotalReward}");
        ImGui.Text($"Estimated total reward {mainArrow.TotalEstimatedReward}");

        ImGui.Checkbox("Show neural network", ref showNeuralNetwork);

        ImGui.End();

        if (showQLearnerNeuralNetwork)
            DrawNeuralNetworkWindow(simulation.QLearner.Network);

        if (showNeuralNetwork)
            DrawNeuralNetworkWindow(simulation.Network);

        if (showRewardGraphs)
            DrawFitnessGraphWindow();
    }

    private static void DrawNeuralNetworkWindow(NeuralNetwork network)
    {
        ImGui.Begin($"Neural Network##{network.GetHashCode()}");
        ImGuiUtils.DisplayNeuralNetwork(network);
        ImGui.End();
    }

    private static void DrawFitnessGraphWindow()
    {
        ImGui.Begin("Reward graphs");

        (List<float>, string)[] graphs = [
            (rewardMedians, nameof(rewardMedians)),
            (estimatedRewardMedians, nameof(estimatedRewardMedians))
        ];

        float minusWidth = 0f;
        foreach ((List<float>, string) tuple in graphs)
            minusWidth = MathF.Max(minusWidth, ImGui.CalcTextSize(tuple.Item2).X);

        DrawGraph(rewardMedians, nameof(rewardMedians), minusWidth);
        DrawGraph(estimatedRewardMedians, nameof(estimatedRewardMedians), minusWidth);

        ImGui.End();
    }

    private static void DrawGraph(List<float> data, string label, float minusWidth)
    {
        ImGui.PlotLines(
            label,
            data,
            data.Count,
            0,
            string.Empty,
            float.MaxValue,
            float.MaxValue,
            new(ImGui.GetContentRegionAvail().X - minusWidth, 50f)
        );
    }

    public static void UpdateRewardGraphsData(Simulation simulation)
    {
        estimatedRewardMedians.Add((float) simulation.Arrows.Median(n => n.TotalEstimatedReward));
        rewardMedians.Add((float) simulation.Arrows.Median(a => a.TotalReward));

        if (estimatedRewardMedians.Count > MaxRewardGraphSize)
            estimatedRewardMedians = estimatedRewardMedians[^MaxRewardGraphSize..];

        if (rewardMedians.Count > MaxRewardGraphSize)
            rewardMedians = rewardMedians[^MaxRewardGraphSize..];
    }
}
