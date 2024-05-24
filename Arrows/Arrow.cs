using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using System;
using MachineLearning;
using MonoGame.Extended;

namespace MonoGameTests
{
    public class Arrow : IComparable<Arrow>
    {
        public static Texture2D Texture;
        public static Vector2 Size = new(50f);

        private const float Velocity = 100f;
        private const float MaxAngleTilting = 0.1f;

        public Vector2 Position { get; private set; }
        public float Angle { get; private set; }

        public readonly NeuralNetwork NeuralNetwork;
        public readonly int CurrentRank;
        public readonly int LastRank;
        
        public Arrow(Vector2 position, Vector2 targetPosition, NeuralNetwork network, int rank, int lastRank)
        {
            Position = position;
            NeuralNetwork = network;
            CurrentRank = rank;
            LastRank = lastRank;
            
            Vector2 windowSize = Application.Instance.WindowSize.ToVector2();
            
            Angle = NeuralNetwork.FeedForward(
                [
                    position.X / windowSize.X,
                    position.Y / windowSize.Y,
                    targetPosition.X / windowSize.X,
                    targetPosition.Y / windowSize.Y
                ]
            )[0];
        }

        public void Update(GameTime gameTime, Vector2 targetPosition)
        {
            Vector2 targetDirection = (targetPosition - Position).NormalizedCopy();
            float targetAngle = MathF.Atan2(targetDirection.Y, targetDirection.X);

            Vector2 windowSize = Application.Instance.WindowSize.ToVector2();
            
            float[] result = NeuralNetwork.FeedForward(
                [
                    Position.X / windowSize.X,
                    Position.Y / windowSize.Y,
                    targetPosition.X / windowSize.X,
                    targetPosition.Y / windowSize.Y
                ]
            );
            float diff = result[0] * MathHelper.TwoPi - Angle;
            Angle += Math.Clamp(diff, -MaxAngleTilting, MaxAngleTilting);

            NeuralNetwork.Fitness += 1f - MathF.Abs(targetAngle - Angle) / MathHelper.TwoPi * 2f;
            const float MaxDistance = 100f;
            NeuralNetwork.Fitness += 1f - MathF.Min(MathF.Abs(targetPosition.X - Position.X), MaxDistance) / MaxDistance;
            NeuralNetwork.Fitness += 1f - MathF.Min(MathF.Abs(targetPosition.Y - Position.Y), MaxDistance) / MaxDistance;

            UpdatePosition(gameTime.GetElapsedSeconds());
        }

        private void UpdatePosition(float deltaTime)
        {
            Position += new Vector2(MathF.Cos(Angle) * Velocity, MathF.Sin(Angle) * Velocity) * deltaTime;
            Position = Vector2.Clamp(Position, Vector2.Zero, Application.Instance.WindowSize.ToVector2());
        }
        
        public void Render(SpriteBatch spriteBatch, Color tintColor)
            => spriteBatch.Draw(Texture, new(Position.ToPoint(), Size.ToPoint()), null, tintColor, Angle, Texture.Bounds.Size.ToVector2() * 0.5f, SpriteEffects.None, 0);

        public int CompareTo(Arrow other)
        {
            if (CurrentRank < other.CurrentRank)
                return -1;

            return CurrentRank > other.CurrentRank ? 1 : 0;
        }
    }
}
