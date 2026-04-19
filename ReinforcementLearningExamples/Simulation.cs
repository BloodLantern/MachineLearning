using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ImGuiNET;
using JetBrains.Annotations;
using MachineLearning.NeuralNetwork;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using MonoGame.Extended;
using MonoGame.Extended.Input;
using MonoGame.Utils;
using MonoGame.Utils.Extensions;

namespace Arrows;

public class Simulation
{
    public const double MaxReward = 100.0;
    private const int ArrowCount = 50;
    private const int NetworkInputCount = 4 + 1 + 1;
    private const int NetworkOutputCount = 1;
    private readonly int[] networkHiddenNeuronsCount = [10, 10];
    private readonly int[] learnerHiddenNeuronsCount = [3, 2];

    private const string SavePath = "network_save.xml";

    private readonly RectangleF arrowSpawnBounds;

    private readonly Random random;
    public Arrow[] Arrows { get; private set; }
    public Arrow MainArrow => Arrows.FirstOrDefault();

    public int CurrentIteration { get; private set; } = 1;

    public double QLearnerGain = 0.1;
    public double QLearnerDiscountFactor = 0.9;
    public NeuralNetwork Network { get; private set; }
    private Episode episode;

    public DeepQLearner QLearner { get; private set; }

    public float NewTimeBetweenResets = 30f;

    public bool Running;
    public bool RunningForOneFrame;

    public float SimulationFrameRate = 300f;
    public bool SimulationSpeedUncapped;

    public Vector2 TargetPosition { get; private set; }
    public Vector2 StartingArrowPosition { get; private set; }
    private const float DeltaTime = 1f / 60f;

    public float TimeBetweenResets { get; private set; }
    public float TimeLeftBeforeReset;
    public int IterationCount => (int) Math.Round(TimeBetweenResets / DeltaTime);

    public bool DrawAllArrows = true;

    private readonly Application application;

    public bool UpdatingQLearner { get; private set; }

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
        QLearner = new(inputCount, learnerHiddenNeuronsCount);
    }

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

    public void RandomizeArrowSpawn() => StartingArrowPosition = GetRandomArrowSpawn();

    public void Update(GameTime gameTime)
    {
        if (UpdatingQLearner)
            return;

        MouseStateExtended mouse = MouseExtended.GetState();
        Vector2 mousePosition = mouse.Position.ToVector2();

        // If the mouse is inside the game window
        if (!ImGui.GetIO().WantCaptureMouse && new Rectangle(Point.Zero, application.WindowSize.ToPoint()).Contains(mouse.Position))
        {
            if (mouse.IsButtonDown(MouseButton.Right))
                TargetPosition = mousePosition;
        }

        if (!Running && !RunningForOneFrame)
            return;

        foreach (Arrow arrow in Arrows)
        {
            arrow.Update(DeltaTime, TargetPosition);

            episode.Iterations.Add(new()
            {
                State = arrow.LastInputs,
                Actions = [arrow.LastOutput],
                Reward = arrow.LastRewardGain,
                EstimatedReward = QLearner.EstimateReward(arrow.LastInputs)
            });
        }

        if (TimeLeftBeforeReset <= 0f)
            ResetSimulation(true);

        TimeLeftBeforeReset -= DeltaTime;

        if (RunningForOneFrame)
            RunningForOneFrame = false;
    }

    public void Draw(SpriteBatch spriteBatch)
    {
        spriteBatch.Begin();

        spriteBatch.DrawCircle(TargetPosition, 10f, 20, Color.Red, 10f);

        spriteBatch.DrawCircle(StartingArrowPosition, 10f, 20, Color.Green, 10f);

        if (DrawAllArrows)
        {
            foreach (Arrow arrow in Arrows)
                arrow.Render(spriteBatch, Color.White);
        }

        MainArrow.Render(spriteBatch, Color.Lime);

        spriteBatch.End();
    }

    public void ResetSimulation(bool evolve)
    {
        SimulationImGui.UpdateRewardGraphsData(this);

        if (evolve)
            EvolveSimulation();

        for (int i = 0; i < Arrows.Length; i++)
        {
            Arrows[i] = new(StartingArrowPosition, this)
            {
                Angle = GetRandomArrowAngle()
            };
        }

        TimeBetweenResets = NewTimeBetweenResets;
        TimeLeftBeforeReset = NewTimeBetweenResets;

        CurrentIteration++;
    }

    private void EvolveSimulation()
    {
        UpdatingQLearner = true;

        Task.Run(() =>
                QLearner.Learn(
                    episode.Iterations.Select(i => new NeuralNetwork.TrainingData(i.State, [i.Reward / MaxReward])).ToArray(),
                    QLearnerGain
                )
            )
            .ContinueWith(_ =>
                {
                    episode.Iterations.Clear();
                    return UpdatingQLearner = false;
                }
            );
    }

    public void SaveNetwork() => Network.Save(SavePath);

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

    public static double ComputeReward(Arrow arrow)
    {
        const float AngleFitnessValueFar = 25f;
        const float MaxAngleValueFar = MathHelper.PiOver2 * 1.5f;

        float result = 0f;
        Vector2 angleDirection = Vector2.FromAngle(arrow.Angle);
        float dot = Vector2.Dot(angleDirection, arrow.TargetDirection);

        result += Calc.ComputeDifference(dot, 1f, MaxAngleValueFar / MathHelper.Pi, AngleFitnessValueFar);

        return result;
    }

    private double EstimateFutureReward(double lastReward, int lastIteration)
    {
        double result = 0.0;

        double lastEstimate = lastReward;
        double discount = 1.0;
        for (int i = lastIteration; i < IterationCount; i++)
        {
            lastEstimate = discount * lastEstimate;
            result += lastEstimate;
            discount *= QLearnerDiscountFactor;
        }

        return result;
    }

    private class Episode
    {
        public readonly List<Iteration> Iterations = [];
    }

    private class Iteration
    {
        public double[] State;
        public double[] Actions;
        public double Reward;
        public double EstimatedReward;
    }
}
