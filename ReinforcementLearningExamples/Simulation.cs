using System;
using System.Linq;
using ImGuiNET;
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
    private const int NetworkInputCount = 3 + 1 + 1;
    private const int NetworkOutputCount = 1;
    private readonly int[] networkHiddenNeuronsCount = [10, 10];
    private readonly int[] learnerHiddenNeuronsCount = [3];

    private const string SavePath = "network_save.xml";

    private readonly RectangleF arrowSpawnBounds;

    private readonly Random random;
    public Arrow[] Arrows { get; private set; }

    public int CurrentIteration { get; private set; } = 1;

    public ActivationFunctionType HiddenLayersActivationFunction = ActivationFunctionType.RectifiedLinearUnit;

    public int NetworkCount { get; } = 100;
    public float NetworkGain = 0.5f;
    public NeuralNetwork[] Networks { get; private set; }

    private QLearner learner;

    public float NewTimeBetweenResets = 30f;
    public ActivationFunctionType OutputLayerActivationFunction = ActivationFunctionType.HyperbolicTangent;

    public bool Running;
    public bool RunningForOneFrame;

    public float SimulationFrameRate = 20f;
    public bool SimulationSpeedUncapped;

    public Vector2 TargetPosition { get; private set; }
    public Vector2 StartingArrowPosition { get; private set; }
    public float StartingArrowAngle { get; private set; }
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
        InitializeNetworks(NetworkCount, NetworkInputCount, NetworkOutputCount);
        TargetPosition = random.NextVector2() * 0.5f * application.WindowSize + application.WindowSize * 0.25f;
        InitializeArrows();
    }

    private void InitializeNetworks(int count, int inputCount, int outputCount)
    {
        Networks = new NeuralNetwork[count];
        learner = new(inputCount, learnerHiddenNeuronsCount);

        NeuralNetwork network = new(random, ComputeFitness, inputCount, outputCount, networkHiddenNeuronsCount);
        Networks[0] = network;

        for (int i = 1; i < count; i++)
        {
            Networks[i] = new(network);
        }
    }

    public const float ArrowAngleEpsilon = 1e-3f;

    private void InitializeArrows()
    {
        RandomizeArrowSpawn();

        Arrows = new Arrow[NetworkCount];

        for (int i = 0; i < Networks.Length; i++)
        {
            Arrows[i] = new(StartingArrowPosition, Networks[i])
            {
                Angle = StartingArrowAngle + ArrowAngleEpsilon * i
            };
        }
    }

    public void RandomizeArrowSpawn()
    {
        StartingArrowPosition = GetRandomArrowSpawn();
        StartingArrowAngle = GetRandomArrowAngle();
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
                arrow.Update(deltaTime, TargetPosition);

            if (TimeLeftBeforeReset <= 0f)
            {
                ResetSimulation(true);
            }
            else
            {
                Array.Sort(Networks);

                for (int i = 0; i < NetworkCount; i++)
                {
                    Networks[i].LearnByGradientDescent(NetworkGain);
                }

                Array.Sort(Arrows);
            }

            TimeLeftBeforeReset -= deltaTime;

            if (RunningForOneFrame)
                RunningForOneFrame = false;
        }
    }

    public void ResetSimulation(bool evolve)
    {
        SimulationImGui.UpdateFitnessGraphsData(this);

        Array.Sort(Networks);

        if (evolve)
            EvolveSimulation();

        for (int i = 0; i < NetworkCount; i++)
        {
            Networks[i].UpdateFitness();
        }

        int selectedArrowIndex = Arrows.IndexOf(SimulationImGui.SelectedArrow);

        StartingArrowAngle = GetRandomArrowAngle();

        for (int i = 0; i < NetworkCount; i++)
        {
            Arrows[i] = new(StartingArrowPosition, Networks[i])
            {
                Angle = StartingArrowAngle + ArrowAngleEpsilon * i
            };

            if (i == selectedArrowIndex)
                SimulationImGui.SelectedArrow = Arrows[i];
        }

        Array.Sort(Arrows);

        TimeBetweenResets = NewTimeBetweenResets;
        TimeLeftBeforeReset = NewTimeBetweenResets;

        CurrentIteration++;
    }

    private void EvolveSimulation()
    {
        const float BestNetworksFraction = 0.5f;
        const float MutateFraction = 0.05f;

        // Assume sorted networks array
        NeuralNetwork[] bestNetworks = Networks[..(int) (NetworkCount / (1f / BestNetworksFraction))];

        // Keep the 50% best and make them learn from their mistakes
        foreach (NeuralNetwork network in bestNetworks)
            network.LearnByGradientDescent(NetworkGain);

        // Then create copies of the best networks and mutate 5% of them
        for (int i = 0; i < NetworkCount; i++)
        {
            Networks[i] = new(bestNetworks[i % bestNetworks.Length]);

            if (i < NetworkCount / (1f / MutateFraction))
                Networks[i].Mutate(random);
        }
    }

    public void SaveBestNetwork() => Arrows[0].Network.Save(SavePath);

    public void LoadSavedNetwork()
    {
        NeuralNetwork saved = NeuralNetwork.Load(SavePath);
        saved.RewardFunction = Networks.First().RewardFunction;

        for (int i = 0; i < NetworkCount; i++)
            Networks[i] = new(saved);

        ResetSimulation(false);
    }

    private Vector2 GetRandomArrowSpawn()
        => random.NextVector2() * ((Point2) arrowSpawnBounds.Size - arrowSpawnBounds.Position) + arrowSpawnBounds.Position;

    private float GetRandomArrowAngle() => random.NextSingle() * MathHelper.TwoPi;

    public const float FitnessBonusMaxDistance = 60f;

    public const float FitnessBonusMaxDistanceSquared = FitnessBonusMaxDistance * FitnessBonusMaxDistance;

    private double ComputeFitness(NeuralNetwork network)
    {
        Arrow arrow = Arrows[Networks.IndexOf(network)];

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
}
