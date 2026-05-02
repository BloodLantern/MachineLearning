using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ImGuiNET;
using JetBrains.Annotations;
using MachineLearning;
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
    public const int ArrowCount = 50;
    public const int NetworkInputCount = 4 + NetworkOutputCount + 1;
    public const int NetworkOutputCount = 2;
    private readonly int[] networkHiddenNeuronsCount = [10, 10];

    private const string SavePath = "network_save.xml";

    private readonly RectangleF arrowSpawnBounds;

    private readonly Random random;
    public Arrow[] Arrows { get; private set; }
    public Arrow MainArrow => Arrows.FirstOrDefault();

    public int CurrentIteration { get; private set; } = 1;

    public double QNetworkGain = 0.1;
    private List<Episode[]> episodes;
    private Episode[] CurrentEpisode => episodes.Last();

    public QNetwork QNetwork { get; private set; }

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
        episodes = [];
        QNetwork = new(random, inputCount, outputCount, networkHiddenNeuronsCount);

        AddNewEpisode();
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

    public void Update()
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

        for (int i = 0; i < Arrows.Length; i++)
        {
            Arrow arrow = Arrows[i];
            arrow.Update(DeltaTime, TargetPosition);

            Episode episode = CurrentEpisode[i];

            if (episode.Iterations.Count > 0)
                episode.Iterations.Last().NextState = arrow.LastInputs;

            // Don't add the current iteration if it is the last one
            if (TimeLeftBeforeReset <= 0f)
                continue;

            episode.Iterations.Add(
                new()
                {
                    State = arrow.LastInputs,
                    Actions = arrow.LastOutputs,
                    Reward = arrow.LastRewardGain
                }
            );
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

        if (CurrentIteration % 5 == 0)
            QNetwork.UpdateTargetNetwork();

        QNetwork.ExplorationProbability = Math.Max(QNetwork.ExplorationProbability * 0.985, 0.1);
        CurrentIteration++;
    }

    private void EvolveSimulation()
    {
        UpdatingQLearner = true;

        Task.Run(() =>
            {
                const int TrainingIterationCount = 50000;
                List<Iteration> trainingData = new(TrainingIterationCount);

                while (trainingData.Count < TrainingIterationCount)
                {
                    Episode episode = episodes.Random(random).Random(random);
                    trainingData.AddRange(random.GetItems(episode.Iterations.ToArray(), random.Next(1, TrainingIterationCount - trainingData.Count)));
                }

                QNetwork.Learn(trainingData.Select(i => (NeuralNetwork.TrainingData) i).ToArray(), QNetworkGain);

                AddNewEpisode();
            })
            .ContinueWith(_ => UpdatingQLearner = false);
    }

    private void AddNewEpisode()
    {
        Episode[] ep = new Episode[ArrowCount];
        for (int i = 0; i < ep.Length; i++)
            ep[i] = new();
        episodes.Add(ep);
    }

    public void SaveNetwork() => QNetwork.Serialize(SavePath);

    public void LoadSavedNetwork()
    {
        QNetwork = QNetwork.Deserialize(SavePath);
        QNetwork.UpdateTargetNetwork();

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

        float currentAngle = arrow.Angle;
        float lastAngle = arrow.Angle - arrow.LastAngleTilting;

        return ComputeAngleReward(currentAngle, arrow.TargetDirection) - ComputeAngleReward(lastAngle, arrow.TargetDirection);

        float ComputeAngleReward(float angle, Vector2 targetDirection)
        {
            Vector2 angleDirection = Vector2.FromAngle(angle);
            float dot = Vector2.Dot(angleDirection, targetDirection);

            return Calc.ComputeDifference(dot, 1f, MaxAngleValueFar / MathHelper.Pi, AngleFitnessValueFar);
        }
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
        public double[] NextState;

        public static explicit operator NeuralNetwork.TrainingData(Iteration iteration)
            => new(iteration.State, iteration.Actions, iteration.Reward, iteration.NextState);
    }
}
