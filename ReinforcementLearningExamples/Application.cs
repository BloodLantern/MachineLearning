using System;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using MonoGame.ImGuiNet;

namespace Arrows;

public class Application : Game
{
    public static Application Instance;

    public GraphicsDeviceManager Graphics { get; }

    private ImGuiRenderer imGuiRenderer;

    private SpriteBatch spriteBatch;

    public int WindowWidth
    {
        get => Graphics.PreferredBackBufferWidth;
        init => Graphics.PreferredBackBufferWidth = value;
    }

    public int WindowHeight
    {
        get => Graphics.PreferredBackBufferHeight;
        init => Graphics.PreferredBackBufferHeight = value;
    }

    public Vector2 WindowSize
    {
        get => new(WindowWidth, WindowHeight);
        init
        {
            WindowWidth = (int) value.X;
            WindowHeight = (int) value.Y;
        }
    }

    public readonly Simulation Simulation;

    public Application()
    {
        Instance = this;
        Graphics = new(this);
        Content.RootDirectory = "Content";
        IsMouseVisible = true;

        Window.AllowUserResizing = true;
        WindowSize = new(1600, 900);

        InactiveSleepTime = TimeSpan.Zero;

        IsFixedTimeStep = false;
        Graphics.SynchronizeWithVerticalRetrace = true;
        Graphics.ApplyChanges();

        Simulation = new(this);
    }

    protected override void Initialize()
    {
        imGuiRenderer = new(this);

        Simulation.Initialize();

        base.Initialize();
    }

    protected override void LoadContent()
    {
        spriteBatch = new(GraphicsDevice);

        imGuiRenderer.RebuildFontAtlas();

        Arrow.Texture = Content.Load<Texture2D>("sssdfg");
    }

    protected override void Update(GameTime gameTime)
    {
        Simulation.Update();

        base.Update(gameTime);
    }

    protected override void Draw(GameTime gameTime)
    {
        GraphicsDevice.Clear(Color.SlateGray);

        Simulation.Draw(spriteBatch);

        base.Draw(gameTime);

        imGuiRenderer.BeginLayout(gameTime);
        SimulationImGui.DrawImGui(Simulation, gameTime);
        imGuiRenderer.EndLayout();
    }

    public void UpdateUncappedFpsState()
    {
        TargetElapsedTime = TimeSpan.FromSeconds(1.0 / Simulation.SimulationFrameRate);
        IsFixedTimeStep = Simulation.SimulationSpeedUncapped;
        Graphics.SynchronizeWithVerticalRetrace = !Simulation.SimulationSpeedUncapped;
        Graphics.ApplyChanges();
    }
}
