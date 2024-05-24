using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using MonoGame.Extended.Input;
using Network;
using System;

namespace MonoGameTests
{
    public class Application : Game
    {
        public static Application Instance;

        private readonly GraphicsDeviceManager graphics;
        private SpriteBatch spriteBatch;

        private const int ArrowCount = 50;
        private readonly Arrow[] arrows = new Arrow[ArrowCount];
        private readonly NeuralNetwork[] networks = new NeuralNetwork[ArrowCount];

        public Random Random = new();

        public int WindowWidth => graphics.PreferredBackBufferWidth;
        public int WindowHeight => graphics.PreferredBackBufferHeight;

        private bool showBest;

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
            for (int i = 0; i < arrows.Length; i++)
            {
                networks[i] = new([2, 10, 10, 1]);
                arrows[i] = new(new(WindowWidth * 0.5f, WindowHeight * 0.5f), 0, networks[i]);
            }

            base.Initialize();
        }

        protected override void LoadContent()
        {
            spriteBatch = new(GraphicsDevice);

            Arrow.Texture = Content.Load<Texture2D>("arrow");
        }

        protected override void Update(GameTime gameTime)
        {
            KeyboardStateExtended keyboard = KeyboardExtended.GetState();

            if (keyboard.WasKeyJustUp(Keys.Space))
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
            }

            if (keyboard.WasKeyJustUp(Keys.B))
                showBest = !showBest;

            foreach (Arrow arrow in arrows)
                arrow.Update(gameTime, targetPosition);

            base.Update(gameTime);
        }

        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice.Clear(Color.Black);

            spriteBatch.Begin();

            if (showBest)
                arrows[0].Render(spriteBatch);
            else
            {
                foreach (Arrow arrow in arrows)
                    arrow.Render(spriteBatch);
            }

            spriteBatch.End();

            base.Draw(gameTime);
        }
    }
}
