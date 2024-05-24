using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using System;
using ImGuiNET;
using MachineLearning;
using MonoGame.Extended;
using MonoGame.Extended.Input;
using MonoGame.ImGuiNet;

namespace MonoGameTests
{
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

        private const int ArrowCount = 50;
        private readonly Arrow[] arrows = new Arrow[ArrowCount];
        private readonly NeuralNetwork[] networks = new NeuralNetwork[ArrowCount];
        
        private const int NetworkInputCount = 4;
        private readonly int[] networkHiddenLayersCount = [10, 10];
        private const int NetworkOutputCount = 1;

        private bool highlightBest;
        private bool resetWorstNeurons;
        
        private Vector2 targetPosition;

        private float timeBetweenResets = 7f;
        private float nextResetTime;

        private int selectedArrowIndex = -1;

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

            InitializeSimulation(0f);
            
            for (int i = 0; i < ArrowCount; i++)
            {
                int[] networkSize = new int[2 + networkHiddenLayersCount.Length];
                networkSize[0] = NetworkInputCount;
                for (int j = 0; j < networkHiddenLayersCount.Length; j++)
                    networkSize[j + 1] = networkHiddenLayersCount[j];
                networkSize[^1] = NetworkOutputCount;
                
                networks[i] = new(networkSize);
                arrows[i] = new(new(WindowWidth * 0.5f, WindowHeight * 0.5f), targetPosition, networks[i], i, -1);
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
            
            foreach (Arrow arrow in arrows)
            {
                arrow.Update(gameTime, targetPosition);

                if (!mouse.WasButtonJustUp(MouseButton.Left))
                    continue;
                
                if (mousePosition.X > arrow.Position.X - Arrow.Size.X &&
                    mousePosition.Y > arrow.Position.Y - Arrow.Size.Y &&
                    mousePosition.X < arrow.Position.X + Arrow.Size.X &&
                    mousePosition.Y < arrow.Position.Y + Arrow.Size.Y)
                    selectedArrowIndex = arrow.CurrentRank;
                else
                    selectedArrowIndex = -1;
            }

            float totalTime = (float) gameTime.TotalGameTime.TotalSeconds;
            if (nextResetTime <= totalTime)
                ResetSimulation(totalTime);

            base.Update(gameTime);
        }

        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice.Clear(Color.SlateGray);

            spriteBatch.Begin();
            
            spriteBatch.DrawCircle(targetPosition, 10f, 20, Color.Red, 10f);

            for (int i = 0; i < ArrowCount; i++)
            {
                Arrow arrow = arrows[i];
                Color color = Color.Lime;
                if (highlightBest)
                {
                    if (i == 0)
                        color = Color.LightYellow;
                    else
                        color *= 0.1f;
                }
                
                arrow.Render(spriteBatch, color);
                
                if (i == selectedArrowIndex)
                    spriteBatch.DrawRectangle(arrow.Position - Arrow.Size * 0.5f, Arrow.Size, Color.Purple, 2f);
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
            
            ImGui.Checkbox("Highlight best", ref highlightBest);
            ImGui.Checkbox("Reset worst neurons", ref resetWorstNeurons);

            ImGui.DragFloat("Time between resets", ref timeBetweenResets, 0.1f, 1f);
            ImGui.TextColored(Color.Orange.ToVector4().ToNumerics(), $"Next reset in {nextResetTime - (float) gameTime.TotalGameTime.TotalSeconds}s");

            if (ImGui.Button("Select best arrow"))
                selectedArrowIndex = 0;

            if (selectedArrowIndex != -1)
            {
                Arrow arrow = arrows[selectedArrowIndex];
                NeuralNetwork network = arrow.NeuralNetwork;
                
                ImGui.SeparatorText("Selected arrow data");
                
                ImGui.Text($"Position {arrow.Position}");
                ImGui.Text($"Angle {arrow.Angle}");
                Vector2 direction = new(MathF.Cos(arrow.Angle), -MathF.Sin(arrow.Angle));
                ImGuiUtils.GridPlotting("Direction", ref direction);
                    
                ImGui.Text($"Current rank {arrow.CurrentRank}");
                ImGui.Text($"Last rank {arrow.LastRank}");
                ImGui.Text($"Fitness {network.Fitness}");
            }
                
            ImGui.End();
        }

        private void ResetSimulation(float totalTime)
        {
            Array.Sort(networks);

            const int HalfArrowCount = ArrowCount / 2;
            for (int i = 0; i < HalfArrowCount; i++)
            {
                networks[HalfArrowCount + i] = new(networks[i]);
                networks[HalfArrowCount + i].Mutate();

                if (resetWorstNeurons)
                    networks[i].ResetNeurons();

                networks[i] = new(networks[i]);
            }

            for (int i = 0; i < ArrowCount; i++)
                networks[i].Rank = i;

            for (int i = 0; i < ArrowCount; i++)
            {
                Arrow arrow = arrows[i];
                
                int rank = 0;
                for (int j = 0; j < ArrowCount; j++)
                {
                    if (arrow.NeuralNetwork != networks[j])
                        continue;
                    
                    rank = j;
                    break;
                }
                
                arrows[i] = new(new(WindowWidth * 0.5f, WindowHeight * 0.5f), targetPosition, networks[i], rank, arrow.CurrentRank);
            }
            
            Array.Sort(arrows);

            InitializeSimulation(totalTime);
        }
        
        private void InitializeSimulation(float totalTime)
        {
            nextResetTime = totalTime + timeBetweenResets;

            Vector2 halfWindowSize = WindowSize.ToVector2() * 0.5f;
            targetPosition = random.NextVector2() * halfWindowSize + halfWindowSize * 0.5f;
        }
    }
}
