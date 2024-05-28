using System;
using MachineLearning;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using MonoGame.Extended;

namespace Arrows;

public class Arrow : IComparable<Arrow>
{
    public static Texture2D Texture;
    public static Vector2 Size = new(50f);

    private const float Velocity = 100f;
    private const float MaxAngleTilting = 0.1f;

    public Vector2 Position { get; private set; }
    public Vector2 TargetDirection { get; private set; }
    public float Angle { get; private set; }
    public float TargetAngle { get; private set; }

    public readonly NeuralNetwork NeuralNetwork;
    public int Rank => NeuralNetwork.Rank;
    public readonly int LastRank;
        
    public Arrow(Vector2 position, Vector2 targetPosition, NeuralNetwork network, int lastRank)
    {
        Position = position;
        NeuralNetwork = network;
        LastRank = lastRank;
        
        TargetDirection = (targetPosition - Position).NormalizedCopy();
        TargetAngle = MathF.Atan2(TargetDirection.Y, TargetDirection.X);
            
        Angle = NeuralNetwork.FeedForward(
            [
                Angle,
                position.X,
                position.Y,
                targetPosition.X,
                targetPosition.Y
            ]
        )[0];
    }

    public void Update(GameTime gameTime, Vector2 targetPosition)
    {
        TargetDirection = (targetPosition - Position).NormalizedCopy();
        TargetAngle = MathF.Atan2(TargetDirection.Y, TargetDirection.X);
            
        float[] result = NeuralNetwork.FeedForward(
            [
                Angle,
                Position.X,
                Position.Y,
                targetPosition.X,
                targetPosition.Y
            ]
        );
        Angle += Math.Clamp(result[0], -MaxAngleTilting, MaxAngleTilting);
        Angle %= MathHelper.TwoPi;

        const float AngleFitnessValue = 10f;
        const float PositionXFitnessValue = 1f;
        const float PositionYFitnessValue = 1f;
        float fitnessDiff = AngleFitnessValue - MathF.Abs(TargetAngle - Angle) / MathHelper.PiOver4 * AngleFitnessValue;
        const float MaxDistance = 200f;
        fitnessDiff += PositionXFitnessValue - MathF.Abs(targetPosition.X - Position.X) / MaxDistance * PositionXFitnessValue;
        fitnessDiff += PositionYFitnessValue - MathF.Abs(targetPosition.Y - Position.Y) / MaxDistance * PositionYFitnessValue;
        
        NeuralNetwork.Fitness += fitnessDiff;

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
        if (Rank < other.Rank)
            return -1;

        return Rank > other.Rank ? 1 : 0;
    }
}
