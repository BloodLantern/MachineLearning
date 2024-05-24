using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Network;
using System;
using ImGuiNET;
using MonoGame.Extended;
using MonoGame.ImGuiNet;

namespace MonoGameTests
{
    public class Application : Game
    {
        public static Application Instance;

        private readonly GraphicsDeviceManager graphics;
        private SpriteBatch spriteBatch;
        private ImGuiRenderer imGuiRenderer;

        private const int ArrowCount = 50;
        private readonly Arrow[] arrows = new Arrow[ArrowCount];
        private readonly NeuralNetwork[] networks = new NeuralNetwork[ArrowCount];

        private readonly Random random = new();

        public int WindowWidth { get => graphics.PreferredBackBufferWidth; set => graphics.PreferredBackBufferWidth = value; }
        public int WindowHeight { get => graphics.PreferredBackBufferHeight; set => graphics.PreferredBackBufferHeight = value; }
        public Point WindowSize => new(WindowWidth, WindowHeight);

        private bool highlightBest;

        private Vector2 targetPosition;

        public Application()
        {
            Instance = this;
            graphics = new(this);
            Content.RootDirectory = "Content";
            IsMouseVisible = true;
        }

        protected override void Initialize()
        {
            imGuiRenderer = new(this);
            
            for (int i = 0; i < arrows.Length; i++)
            {
                networks[i] = new([2, 10, 10, 1]);
                arrows[i] = new(new(WindowWidth * 0.5f, WindowHeight * 0.5f), 0, networks[i]);
            }

            ChangeTargetPosition();

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
            foreach (Arrow arrow in arrows)
                arrow.Update(gameTime, targetPosition);

            base.Update(gameTime);
        }

        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice.Clear(Color.Black);

            spriteBatch.Begin();
            
            spriteBatch.DrawPoint(targetPosition, Color.Red, 10f);

            for (int i = 0; i < arrows.Length; i++)
            {
                Arrow arrow = arrows[i];
                if (highlightBest && i != 0)
                    arrow.Render(spriteBatch, Color.White * 0.5f);
                else
                    arrow.Render(spriteBatch);
            }

            spriteBatch.End();

            base.Draw(gameTime);

            imGuiRenderer.BeginLayout(gameTime);
            DrawImGui(gameTime);
            imGuiRenderer.EndLayout();
        }

        protected virtual void DrawImGui(GameTime gameTime)
        {
            ImGui.Begin("Simulation settings");
            
            if (ImGui.Button("Reset"))
                ResetSimulation();
            
            ImGui.Checkbox("Highlight best", ref highlightBest);
            
            ImGui.End();
        }

        private void ResetSimulation()
        {
            Array.Sort(networks);

            for (int i = 0; i < ArrowCount / 2; i++)
            {
                networks[ArrowCount / 2 + i] = new(networks[i]);
                networks[ArrowCount / 2 + i].Mutate();

                //Networks[i].ResetNeurons();
                networks[i] = new(networks[i]);
            }

            for (int i = 0; i < ArrowCount; i++)
            {
                arrows[i] = new(new(WindowWidth * 0.5f, WindowHeight * 0.5f), MathHelper.PiOver2, networks[i]);
            }

            ChangeTargetPosition();
        }
        
        private void ChangeTargetPosition() => targetPosition = random.NextVector2() * WindowSize.ToVector2();
    }
}
