interface CloudLayerProps {
  planetRadius: number;
  /** Deck altitude above the reference radius, in scene units. */
  deckAltitude: number;
  /** Shell opacity 0..1 from the cloud mapping. */
  opacity: number;
  /** False when the retrieval left the cloud-top pressure unconstrained. */
  constrained: boolean;
}

/**
 * Generic cloud/haze deck: a single semi-transparent shell, uniform in all
 * directions. When unconstrained it renders as a wireframe overlay and the
 * panel labels it uncertain — never a cloud map.
 */
export function CloudLayer({ planetRadius, deckAltitude, opacity, constrained }: CloudLayerProps) {
  const radius = planetRadius + deckAltitude;
  return (
    <group>
      <mesh>
        <sphereGeometry args={[radius, 48, 48]} />
        <meshBasicMaterial color="#e5e7eb" transparent opacity={opacity} depthWrite={false} />
      </mesh>
      {!constrained && (
        <mesh>
          <sphereGeometry args={[radius * 1.002, 24, 16]} />
          <meshBasicMaterial color="#9ca3af" wireframe transparent opacity={0.35} depthWrite={false} />
        </mesh>
      )}
    </group>
  );
}
