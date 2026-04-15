using System;
using System.Collections.Generic;
using System.Linq;
using ImGuiNET;
using JetBrains.Annotations;
using MachineLearning;
using MachineLearning.Models.NeuralNetwork;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using MonoGame.Extended;
using MonoGame.Extended.Input;
using MonoGame.Utils;
using MonoGame.Utils.Extensions;

namespace Arrows;

public class Simulation
{
    private const int ArrowCount = 100;
    private const int NetworkInputCount = 4 + 1 + 1;
    private const int NetworkOutputCount = 1;
    private readonly int[] networkHiddenNeuronsCount = [10, 10];
    private readonly int[] learnerHiddenNeuronsCount = [3];

    private const string SavePath = "network_save.xml";

    private readonly RectangleF arrowSpawnBounds;

    private readonly Random random;
    public Arrow[] Arrows { get; private set; }

    public int CurrentIteration { get; private set; } = 1;

    public ActivationFunctionType HiddenLayersActivationFunction = ActivationFunctionType.RectifiedLinearUnit;

    public float QLearnerGain = 0.5f;
    public NeuralNetwork Network { get; private set; }
    private Episode episode;

    private QLearner qLearner;

    public float NewTimeBetweenResets = 30f;
    public ActivationFunctionType OutputLayerActivationFunction = ActivationFunctionType.Sigmoid;

    public bool Running;
    public bool RunningForOneFrame;

    public float SimulationFrameRate = 20f;
    public bool SimulationSpeedUncapped;

    public Vector2 TargetPosition { get; private set; }
    public Vector2 StartingArrowPosition { get; private set; }
    private float deltaTime;

    public float TimeBetweenResets { get; private set; }
    public float TimeLeftBeforeReset;

    private readonly Application application;

    public Simulation(Application application)
    {
        this.application = application;

        random = new(DateTime.Now.Millisecond);

        arrowSpawnBounds = new(application.WindowSize * 0.1f, application.WindowSize * 0.9f);

        TimeLeftBeforeReset = NewTimeBetweenResets;
    }

    public void Initialize()
    {
        InitializeNetworks(NetworkInputCount, NetworkOutputCount);
        TargetPosition = random.NextVector2() * 0.5f * application.WindowSize + application.WindowSize * 0.25f;
        InitializeArrows();
    }

    private void InitializeNetworks(int inputCount, int outputCount)
    {
        Network = new(random, inputCount, outputCount, networkHiddenNeuronsCount);
        episode = new();
        qLearner = new(inputCount, learnerHiddenNeuronsCount);
    }

    public const float ArrowAngleEpsilon = 1e-3f;

    private void InitializeArrows()
    {
        RandomizeArrowSpawn();

        Arrows = new Arrow[ArrowCount];

        for (int i = 0; i < Arrows.Length; i++)
        {
            Arrows[i] = new(StartingArrowPosition, this)
            {
                Angle = GetRandomArrowAngle()
            };
        }
    }

    public void RandomizeArrowSpawn()
    {
        StartingArrowPosition = GetRandomArrowSpawn();
        GetRandomArrowAngle();
    }

    public void Update(GameTime gameTime)
    {
        MouseStateExtended mouse = MouseExtended.GetState();
        Vector2 mousePosition = mouse.Position.ToVector2();

        deltaTime = SimulationSpeedUncapped ? 1f / SimulationFrameRate : gameTime.GetElapsedSeconds();

        // If mouse is inside the game window
        if (!ImGui.GetIO().WantCaptureMouse &&
            new Rectangle(Point.Zero, application.WindowSize.ToPoint()).Contains(mouse.Position))
        {
            if (mouse.IsButtonDown(MouseButton.Right))
                TargetPosition = mousePosition;

            if (mouse.WasButtonJustDown(MouseButton.Left))
            {
                foreach (Arrow arrow in Arrows)
                {
                    if ((mousePosition - arrow.Position).Length() < Arrow.Size.X * 0.5f)
                    {
                        SimulationImGui.SelectedArrow = arrow;
                        break;
                    }

                    SimulationImGui.SelectedArrow = null;
                }
            }
        }

        if (Running || RunningForOneFrame)
        {
            foreach (Arrow arrow in Arrows)
            {
                arrow.Update(deltaTime, TargetPosition);

                episode.Iterations.Add(new()
                {
                    State = arrow.LastInputs,
                    Actions = [arrow.LastOutput],
                    Reward = arrow.LastRewardGain
                });
            }

            if (TimeLeftBeforeReset <= 0f)
                ResetSimulation(true);
            else
                Arrows.Sort();

            TimeLeftBeforeReset -= deltaTime;

            if (RunningForOneFrame)
                RunningForOneFrame = false;
        }
    }

    public void ResetSimulation(bool evolve)
    {
        SimulationImGui.UpdateRewardGraphsData(this);

        if (evolve)
            EvolveSimulation();

        int selectedArrowIndex = Arrows.IndexOf(SimulationImGui.SelectedArrow);

        for (int i = 0; i < Arrows.Length; i++)
        {
            Arrows[i] = new(StartingArrowPosition, this)
            {
                Angle = GetRandomArrowAngle()
            };

            if (i == selectedArrowIndex)
                SimulationImGui.SelectedArrow = Arrows[i];
        }

        TimeBetweenResets = NewTimeBetweenResets;
        TimeLeftBeforeReset = NewTimeBetweenResets;

        CurrentIteration++;
    }

    private void EvolveSimulation()
    {
        foreach (Iteration episodeIteration in episode.Iterations)
        {
            qLearner.LearnByGradientDescent(episodeIteration.State, episodeIteration.Reward, QLearnerGain);
        }

        SimulationImGui.SelectedArrow = Arrows.First();
    }

    public void SaveNetwork() => Arrows.First().Network.Save(SavePath);

    public void LoadSavedNetwork()
    {
        Network = NeuralNetwork.Load(SavePath);

        ResetSimulation(false);
    }

    [MustUseReturnValue]
    private Vector2 GetRandomArrowSpawn()
        => random.NextVector2() * ((Point2) arrowSpawnBounds.Size - arrowSpawnBounds.Position) + arrowSpawnBounds.Position;

    [MustUseReturnValue]
    private float GetRandomArrowAngle() => random.NextSingle() * MathHelper.TwoPi;

    public double ComputeReward(Arrow arrow)
    {
        const float AngleFitnessValueFar = 25f;
        const float MaxAngleValueFar = MathHelper.PiOver2 * 1.5f;

        float result = 0f;
        Vector2 angleDirection = Vector2.FromAngle(arrow.Angle);
        float dot = Vector2.Dot(angleDirection, arrow.TargetDirection);

        result += Calc.ComputeDifference(dot, 1f, MaxAngleValueFar / MathHelper.Pi, AngleFitnessValueFar);

        return result * deltaTime;
    }

    public void Draw(SpriteBatch spriteBatch)
    {
        spriteBatch.Begin();

        spriteBatch.DrawCircle(TargetPosition, 10f, 20, Color.Red, 10f);

        spriteBatch.DrawCircle(StartingArrowPosition, 10f, 20, Color.Green, 10f);

        foreach (Arrow arrow in Arrows)
        {
            Color color = Color.White;
            if (SimulationImGui.SelectedArrow != null && SimulationImGui.SelectedArrow != arrow)
                color *= 0.05f;

            arrow.Render(spriteBatch, color);
        }

        SimulationImGui.SelectedArrow?.Render(spriteBatch, Color.Lime);

        spriteBatch.End();
    }

    private class Episode
    {
        public List<Iteration> Iterations = [];
    }

    private class Iteration
    {
        public double[] State;
        public double[] Actions;
        public double Reward;
    }
}
