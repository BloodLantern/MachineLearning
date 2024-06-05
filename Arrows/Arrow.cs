using System;
using MachineLearning.Models.NeuralNetwork;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using MonoGame.Extended;
using MonoGame.Utils;

namespace Arrows;

public class Arrow : IComparable<Arrow>
{
    public static Texture2D Texture;
    public static Vector2 Size = new(50f);

    private const float Velocity = 100f;
    public const float MaxAngleTilting = 0.1f;

    public Vector2 Position { get; set; }
    public Vector2 TargetDirection { get; private set; }

    private float angle;
    public float Angle
    {
        get => angle;
        set => angle = Calc.ClampRadiantAngle(value);
    }
    private float targetAngle;
    public float TargetAngle
    {
        get => targetAngle;
        private set => targetAngle = Calc.ClampRadiantAngle(value);
    }

    public readonly NeuralNetwork Network;
    public int Rank => Network.Rank;
    public readonly int LastRank;
        
    public Arrow(Vector2 position, NeuralNetwork network, int lastRank)
    {
        Position = position;
        Network = network;
        LastRank = lastRank;
    }

    public void Update(float deltaTime, Vector2 targetPosition)
    {
        Network.UpdateFitness();
        Angle += ComputeNextAngleDelta(targetPosition);

        UpdatePosition(deltaTime);
    }

    private void UpdatePosition(float deltaTime)
    {
        Position += new Vector2(MathF.Cos(Angle) * Velocity, MathF.Sin(Angle) * Velocity) * deltaTime;
        Position = Vector2.Clamp(Position, Vector2.Zero, Application.Instance.WindowSize.ToVector2());
    }

    public float ComputeNextAngleDelta(Vector2 targetPosition)
    {
        TargetDirection = (targetPosition - Position).NormalizedCopy();
        TargetAngle = MathF.Atan2(TargetDirection.Y, TargetDirection.X);
        
        Vector2 windowSize = Application.Instance.WindowSize.ToVector2();
        double[] networkOutput = Network.ComputeOutputs(
            [
                Angle / MathHelper.TwoPi,
                Position.X / windowSize.X,
                Position.Y / windowSize.Y,
                targetPosition.X / windowSize.X,
                targetPosition.Y / windowSize.Y,
                Application.Instance.TimeLeftBeforeReset / Application.Instance.TimeBetweenResets
            ]
        );

        return (float) networkOutput[0] * MaxAngleTilting;
    }
        
    public void Render(SpriteBatch spriteBatch, Color tintColor)
        => spriteBatch.Draw(Texture, new(Position.ToPoint(), Size.ToPoint()), null, tintColor, Angle, Texture.Bounds.Size.ToVector2() * 0.5f, SpriteEffects.None, 0f);

    public int CompareTo(Arrow other)
    {
        if (Rank < other.Rank)
            return -1;

        return Rank > other.Rank ? 1 : 0;
    }
}
