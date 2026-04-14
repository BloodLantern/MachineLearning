using System;
using System.Linq;
using MachineLearning;
using MachineLearning.Models.NeuralNetwork;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using MonoGame.Extended;
using MonoGame.Utils;
using MonoGame.Utils.Extensions;

namespace Arrows;

public class Arrow
{
    private const float Speed = 100f;
    public const float MaxAngleTilting = MathHelper.TwoPi * 2f;

    public static Texture2D Texture;
    public static Vector2 Size = new(50f);

    public readonly NeuralNetwork Network;

    public Vector2 Position { get; set; }

    public Vector2 Direction { get; private set; }

    public Vector2 TargetDirection { get; private set; }

    public float Angle { get; set; }

    public float TargetAngle { get; private set; }

    public float LastAngleTilting { get; private set; }

    public bool AngleFlipped { get; private set; }

    private double lastOutput;
    private double lastFitnessGain;

    public Arrow(Vector2 position, NeuralNetwork network)
    {
        Position = position;
        Network = network;
    }

    public void Update(float deltaTime, Vector2 targetPosition)
    {
        UpdatePosition(deltaTime);

        LastAngleTilting = ComputeNextAngleDelta(targetPosition) * deltaTime;

        Angle += LastAngleTilting;
        Direction = Vector2.FromAngle(Angle);

        lastFitnessGain = Network.UpdateFitness();
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

        double[] networkOutputs = Network.ComputeOutputs(
            [
                (Vector2.Dot(Direction, TargetDirection) + 1f) * 0.5f, // Difference with the target angle
                Utils.BoolToFloat(difference.LengthSquared() < Application.FitnessBonusMaxDistanceSquared), // Distance to the target
                Utils.BoolToFloat(AngleFlipped), // Whether the arrow's angle delta is flipped
                lastOutput,
                lastFitnessGain
            ],
            ActivationFunctions.GetFromType(Application.Instance.HiddenLayersActivationFunction),
            ActivationFunctions.GetFromType(Application.Instance.OutputLayerActivationFunction)
        );

        double singleOutput = networkOutputs.Single();
        lastOutput = singleOutput;
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
}
