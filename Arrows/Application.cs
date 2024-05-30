using System;
using ImGuiNET;
using MachineLearning.Models.NeuralNetwork;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using MonoGame.Extended;
using MonoGame.Extended.Input;
using MonoGame.ImGuiNet;

namespace Arrows;

public class Application : Game
{
    public static Application Instance;

    private readonly GraphicsDeviceManager graphics;
    private SpriteBatch spriteBatch;
    private ImGuiRenderer imGuiRenderer;

    public int WindowWidth { get => graphics.PreferredBackBufferWidth; init => graphics.PreferredBackBufferWidth = value; }
    public int WindowHeight { get => graphics.PreferredBackBufferHeight; init => graphics.PreferredBackBufferHeight = value; }
    public Point WindowSize => new(WindowWidth, WindowHeight);

    private readonly Random random = new();

    private const int ArrowCount = 200;
    private readonly Arrow[] arrows = new Arrow[ArrowCount];
    private readonly NeuralNetwork[] networks = new NeuralNetwork[ArrowCount];
        
    private const int NetworkInputCount = 5;
    private readonly int[] networkHiddenLayersCount = [20, 20, 20, 20];
    private const int NetworkOutputCount = 1;

    private const string SavePath = "network_save.xml";

    private bool running;
    private bool runningForOneFrame;
        
    private Vector2 targetPosition;

    private float timeBetweenResets = 7f;
    private float timeLeftBeforeReset;
    private int currentIteration;
    
    private bool bestArrowSelected;
    private Arrow selectedArrow;

    public Application()
    {
        Instance = this;
        graphics = new(this);
        Content.RootDirectory = "Content";
        IsMouseVisible = true;

        WindowWidth = 1280;
        WindowHeight = 720;
    }

    protected override void Initialize()
    {
        imGuiRenderer = new(this);

        InitializeSimulation();
            
        for (int i = 0; i < ArrowCount; i++)
        {
            int[] networkSize = new int[2 + networkHiddenLayersCount.Length];
            networkSize[0] = NetworkInputCount;
            for (int j = 0; j < networkHiddenLayersCount.Length; j++)
                networkSize[j + 1] = networkHiddenLayersCount[j];
            networkSize[^1] = NetworkOutputCount;
                
            networks[i] = new(networkSize)
            {
                Rank = i
            };
            arrows[i] = new(GetRandomPosition(), targetPosition, networks[i], -1);
        }

        base.Initialize();
    }

    protected override void LoadContent()
    {
        spriteBatch = new(GraphicsDevice);

        imGuiRenderer.RebuildFontAtlas();

        Arrow.Texture = Content.Load<Texture2D>("arrow");
    }

    protected override void Update(GameTime gameTime)
    {
        MouseStateExtended mouse = MouseExtended.GetState();
        Vector2 mousePosition = mouse.Position.ToVector2();

        if (mouse.WasButtonJustUp(MouseButton.Left) && !ImGui.GetIO().WantCaptureMouse)
        {
            foreach (Arrow arrow in arrows)
            {
                if (mousePosition.X > arrow.Position.X - Arrow.Size.X * 0.5f &&
                    mousePosition.Y > arrow.Position.Y - Arrow.Size.Y * 0.5f &&
                    mousePosition.X < arrow.Position.X + Arrow.Size.X * 0.5f &&
                    mousePosition.Y < arrow.Position.Y + Arrow.Size.Y * 0.5f)
                {
                    selectedArrow = arrow;
                    break;
                }
                
                selectedArrow = null;
            }
        }

        if (running || runningForOneFrame)
        {
            foreach (Arrow arrow in arrows)
                arrow.Update(gameTime, targetPosition);

            if (timeLeftBeforeReset <= 0f)
            {
                ResetSimulation();
            }
            else
            {
                Array.Sort(networks);

                for (int i = 0; i < ArrowCount; i++)
                    networks[i].Rank = i;
                
                Array.Sort(arrows);
            }

            timeLeftBeforeReset -= gameTime.GetElapsedSeconds();

            if (runningForOneFrame)
                runningForOneFrame = false;
        }

        base.Update(gameTime);
    }

    protected override void Draw(GameTime gameTime)
    {
        GraphicsDevice.Clear(Color.SlateGray);

        spriteBatch.Begin();
            
        spriteBatch.DrawCircle(targetPosition, 10f, 20, Color.Red, 10f);

        foreach (Arrow arrow in arrows)
        {
            Color color = Color.Blue;
            if (selectedArrow != null)
            {
                if (arrow == selectedArrow)
                    color = Color.Lime;
                else
                    color *= 0.05f;
            }
                
            arrow.Render(spriteBatch, color);
        }

        spriteBatch.End();

        base.Draw(gameTime);

        imGuiRenderer.BeginLayout(gameTime);
        DrawImGui(gameTime);
        imGuiRenderer.EndLayout();
    }

    protected virtual void DrawImGui(GameTime gameTime)
    {
        ImGui.Begin("Simulation");
            
        ImGui.SeparatorText("Settings");
            
        ImGui.DragFloat("Time between resets", ref timeBetweenResets, 0.1f, 1f);
        
        ImGui.SeparatorText("Readonly data");
        
        ImGui.Text($"FPS: {1f / gameTime.GetElapsedSeconds()}");
        ImGui.Text($"Total time: {gameTime.TotalGameTime}");
        ImGui.Text($"Current iteration: {currentIteration}");
        
        ImGui.TextColored(Color.Orange.ToVector4().ToNumerics(), $"Next reset in {timeLeftBeforeReset}s");
        ImGui.Text($"Target position {targetPosition}");
            
        ImGui.SeparatorText("Actions");

        ImGui.Checkbox("Running", ref running);

        if (ImGui.Button("Advance one generation"))
            ResetSimulation();

        if (ImGui.Button("Advance one frame"))
            runningForOneFrame = true;

        if (ImGui.Button("Save best"))
            SaveBestNetwork();

        if (ImGui.Button("Load save"))
            LoadSavedNetwork();

        if (ImGui.Button("Select best arrow"))
            selectedArrow = arrows[0];

        if (ImGui.Checkbox("Keep best arrow selected", ref bestArrowSelected) && !bestArrowSelected)
            selectedArrow = null;

        if (bestArrowSelected)
            selectedArrow = arrows[0];

        if (selectedArrow != null)
        {
            NeuralNetwork network = selectedArrow.NeuralNetwork;
                
            ImGui.SeparatorText("Selected arrow data");
                
            ImGui.Text($"Position {selectedArrow.Position}");
            ImGui.Text($"Angle {selectedArrow.Angle}rad = {MathHelper.ToDegrees(selectedArrow.Angle)}deg");
            Vector2 direction = new(MathF.Cos(selectedArrow.Angle), -MathF.Sin(selectedArrow.Angle));
            ImGuiUtils.DirectionVector("Direction", ref direction, selectedArrow.TargetDirection);
                    
            ImGui.Text($"Current rank {selectedArrow.Rank}");
            ImGui.Text($"Last rank {selectedArrow.LastRank}");
            ImGui.Text($"Fitness {network.Fitness}");
        }
                
        ImGui.End();
    }

    private void ResetSimulation()
    {
        Array.Sort(networks);

        // Mutate the worst 50%
        const float MutatedArrowRatio = 0.5f;
        const int MutatedArrowCount = (int) (ArrowCount * MutatedArrowRatio);
        const int NonMutatedArrowCount = ArrowCount - MutatedArrowCount;
        for (int i = 0; i < MutatedArrowCount; i++)
        {
            NeuralNetwork goodNetwork = networks[i];
            
            networks[NonMutatedArrowCount + i] = new(goodNetwork);
            
            NeuralNetwork badNetwork = networks[NonMutatedArrowCount + i];
            
            badNetwork.Mutate();
        }

        for (int i = 0; i < ArrowCount; i++)
        {
            networks[i].Rank = i;
            networks[i].Fitness = 0.0;
        }

        int selectedArrowIndex = selectedArrow?.Rank ?? -1;

        for (int i = 0; i < ArrowCount; i++)
            arrows[i] = new(GetRandomPosition(), targetPosition, networks[i], arrows[i].Rank);

        if (selectedArrowIndex != -1)
            selectedArrow = arrows[selectedArrowIndex];
        
        Array.Sort(arrows);

        InitializeSimulation();
    }

    private void InitializeSimulation()
    {
        timeLeftBeforeReset = timeBetweenResets;

        Vector2 halfWindowSize = WindowSize.ToVector2() * 0.5f;
        targetPosition = random.NextVector2() * halfWindowSize + halfWindowSize * 0.5f;

        currentIteration++;
    }

    private void SaveBestNetwork() => arrows[0].NeuralNetwork.Save(SavePath);

    private void LoadSavedNetwork()
    {
        for (int i = 0; i < arrows.Length; i++)
            arrows[i] = new(GetRandomPosition(), targetPosition, NeuralNetwork.Load(SavePath), -1);
        
        ResetSimulation();
    }

    private Vector2 GetRandomPosition() => new(random.NextSingle() * WindowWidth, random.NextSingle() * WindowHeight);
}
