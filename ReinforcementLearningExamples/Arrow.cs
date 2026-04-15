using System;
using System.Linq;
using MachineLearning;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using MonoGame.Extended;
using MonoGame.Utils.Extensions;

namespace Arrows;

public class Arrow : IComparable<Arrow>
{
    private const float Speed = 100f;
    public const float MaxAngleTilting = MathHelper.TwoPi * 2f;

    public static Texture2D Texture;
    public static Vector2 Size = new(50f);

    public Vector2 Position { get; set; }

    public Vector2 Direction { get; private set; }

    public Vector2 TargetDirection { get; private set; }

    public float Angle { get; set; }

    public float TargetAngle { get; private set; }

    public float LastAngleTilting { get; private set; }

    public double[] LastInputs { get; private set; }
    public double LastOutput { get; private set; }
    public double LastRewardGain { get; private set; }
    public double LastEstimatedRewardGain { get; private set; }

    public double TotalReward { get; private set; }
    public double TotalEstimatedReward { get; private set; }

    private readonly Simulation simulation;

    public Arrow(Vector2 position, Simulation simulation)
    {
        Position = position;
        this.simulation = simulation;
    }

    public void Update(float deltaTime, Vector2 targetPosition)
    {
        UpdatePosition(deltaTime);

        LastAngleTilting = ComputeNextAngleDelta(targetPosition) * deltaTime;

        Angle += LastAngleTilting;
        Direction = Vector2.FromAngle(Angle);

        LastRewardGain = simulation.Network.ComputeRewardGain(_ => Simulation.ComputeReward(this));
        LastEstimatedRewardGain = simulation.QLearner.EstimateQuality(LastInputs) * 100;

        TotalReward += LastRewardGain;
        TotalEstimatedReward += LastEstimatedRewardGain;
    }

    private void UpdatePosition(float deltaTime)
    {
        Position += Direction * (Speed * deltaTime);
        Position = Vector2.Clamp(Position, Vector2.Zero, Application.Instance.WindowSize);
    }

    public float ComputeNextAngleDelta(Vector2 targetPosition)
    {
        Vector2 difference = targetPosition - Position;
        TargetDirection = difference.NormalizedCopy();
        TargetAngle = MathF.Atan2(TargetDirection.Y, TargetDirection.X) - MathHelper.Pi;

        LastInputs = [
            Direction.X,
            Direction.Y,
            TargetDirection.X,
            TargetDirection.Y,

            LastOutput,
            LastRewardGain / 100.0
        ];
        double[] networkOutputs = simulation.Network.ComputeOutputs(
            LastInputs,
            ActivationFunctions.GetFromType(simulation.HiddenLayersActivationFunction),
            ActivationFunctions.GetFromType(simulation.OutputLayerActivationFunction)
        );

        double singleOutput = networkOutputs.Single();
        LastOutput = singleOutput;
        return (float) Math.Clamp(singleOutput, -1.0, 1.0) * MaxAngleTilting;
    }

    public void Render(SpriteBatch spriteBatch, Color tintColor) => spriteBatch.Draw(
        Texture,
        new(Position.ToPoint(), Size.ToPoint()),
        null,
        tintColor,
        Angle,
        Texture.Bounds.Size.ToVector2() * 0.5f,
        SpriteEffects.None,
        0f
    );

    public int CompareTo(Arrow other) => TotalReward.CompareTo(other.TotalReward);
}
