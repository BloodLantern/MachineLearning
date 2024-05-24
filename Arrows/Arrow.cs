using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Network;
using System;

namespace MonoGameTests
{
    public class Arrow
    {
        public static Texture2D Texture;

        private const float Velocity = 100f;
        private static Vector2 size = new(50);

        private Vector2 position;
        private float angle;

        private readonly NeuralNetwork NeuralNetwork;
        
        public Arrow(Vector2 position, float angle, NeuralNetwork network)
        {
            this.position = position;
            this.angle = angle;
            NeuralNetwork = network;
        }

        public void Update(GameTime gameTime, Vector2 targetPosition)
        {
            Vector2 targetDirection = targetPosition - position;
            float targetAngle = (float) Math.Atan2(targetDirection.Y, targetDirection.X);

            float[] result = NeuralNetwork.FeedForward([angle / MathHelper.TwoPi, targetAngle]);
            angle = result[0] * MathHelper.TwoPi;

            position += new Vector2(MathF.Cos(angle) * Velocity, MathF.Sin(angle) * Velocity) * (float) gameTime.ElapsedGameTime.TotalSeconds;
            position = Vector2.Clamp(position, Vector2.Zero, new(Application.Instance.WindowWidth, Application.Instance.WindowHeight));

            NeuralNetwork.Fitness += 1f - MathF.Abs(targetAngle - angle) / MathHelper.TwoPi;
        }

        public void Render(SpriteBatch spriteBatch) => Render(spriteBatch, Color.White);

        public void Render(SpriteBatch spriteBatch, Color tintColor)
        {
            spriteBatch.Draw(Texture, new(position.ToPoint(), size.ToPoint()), null, tintColor, angle, Texture.Bounds.Size.ToVector2() * 0.5f, SpriteEffects.None, 0);
        }
    }
}
