/**
 * DEAD TREES for the RATFIRE grasslands — four gnarled leafless trees
 * separated out of the uploaded "dead_trees.glb" (each of its four meshes
 * is one complete tree), streamed over the endless terrain with the
 * SAME world-anchored recycling scheme as the other decor layers:
 *
 *  - The world is cut into one-block cells; every instance owns a fixed
 *    cell residue inside a (2R+1)² window around the player's cell, so
 *    walking any direction keeps fresh dead trees materialising ahead
 *    while the same patch of land always regrows the SAME trees
 *    (deterministic per-cell hashes pick variant, size, tilt, rotation
 *    and jitter — zero allocation per frame).
 *  - Cells are rejected when too steep (cliff edges) or inside the spawn
 *    keep-clear disc (the lobby showcase stays unobstructed).
 *
 * RENDERING — one InstancedMesh per tree variant (4 draw calls total).
 * The GLB materials are flat greys with no images, so every variant is
 * painted with the SAME pixel bark texture as the stumps
 * (public/models/trunk_bark.png) using generated box-projection UVs:
 * each vertex projects along its dominant normal axis, which keeps the
 * texel density even across trunks, forks and dead branches alike (the
 * model's own UVs are Maya-style atlases that would smear the bark).
 *
 * SOLID COLLISION — dead trees block the player like the stumps: the
 * handle exposes pushOut(x, z, radius). The collider is a circle per
 * tree whose radius is baked from the variant's REAL ground-level
 * silhouette (root flare band), scaled per instance — a natural
 * wall-slide, cheap and tunnel-free at player speeds.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

export interface DeadTreeSystemOptions {
  /** World units per voxel block edge (100). */
  block: number;
  /** Grid origin offset (64) — matches terrainChunks' grid mapping. */
  gridOffset: number;
  /** Terrain surface height at a world position (top of the block). */
  surfaceYAt(worldX: number, worldZ: number): number;
  /** Deterministic integer block height at grid coordinates. */
  blockHeight(gx: number, gz: number): number;
  /** Mobile LOW tier: fewer trees, no shadow casting. */
  lowSpec?: boolean;
  /** Keep-clear disc around the player spawn (lobby showcase). */
  clearX?: number;
  clearZ?: number;
  clearRadius?: number;
}

export interface DeadTreeSystemHandle {
  /** True once the model finished loading and instances exist. */
  ready(): boolean;
  /** Recycle cells around the player. Call per frame (cheap no-op when
   *  the player hasn't crossed a block boundary). */
  update(playerX: number, playerZ: number): void;
  /** Slide a collider of `radius` out of every dead tree it overlaps.
   *  Returns the corrected position (same object every call). */
  pushOut(x: number, z: number, radius: number): { x: number; z: number };
  stats(): { instances: number; placed: number };
  dispose(): void;
}

/* ------------------------------ tuning ------------------------------ */

/** The uploaded Sketchfab dead-trees pack — four separate tree meshes
 *  in one scene, laid out side by side. */
const DEAD_TREES_URL = '/models/dead_trees.glb';

/** Pixel bark shared with the stumps — wraps every trunk and branch. */
const DEAD_BARK_URL = '/models/trunk_bark.png';

/** Dead trees per one-BLOCK² cell (cell = 100 world units). Sparse:
 *  lone skeletal silhouettes, not a forest. Desktop window holds
 *  (2·40+1)² cells → ~29 dead trees around the traveller. */
const DENSITY = 0.0045;
const DENSITY_LOW = 0.0032;

/** Wrap window half-width, in cells (matches the other decor layers). */
const RADIUS_DESKTOP = 40;
const RADIUS_LOW = 26;

/** Stand-on-its-own column jitter (fraction of a block). */
const JITTER_FRACTION = 0.72;

/** Max block-height range across a cell before it counts as a cliff. */
const SLOPE_MAX_BLOCKS = 2;

/** Height range in world units BEFORE the per-variant multiplier — the
 *  character is 190 tall, so dead trees tower ~1.3×..2.8× over him. */
const HEIGHT_MIN = 295;
const HEIGHT_VAR = 165;

/** Per-variant height multipliers for the four GLB trees (keeps the
 *  pack's own size ranking: tree #0 is the runt, tree #3 the giant). */
const VARIANT_HEIGHT_MULT = [0.85, 0.95, 1.05, 1.15];

/** How deep each tree roots into its block: a fraction of its height
 *  with a floor so even young snags hug the terrain. */
const SINK_FRACTION = 0.05;
const SINK_MIN = 6;

/** Dead trees lean further than fresh stumps (radians). */
const TILT_MAX = 0.07;

/** Fraction of the variant's unit height whose vertices define the
 *  collision radius (the root-flare band). */
const BASE_BAND = 0.08;

/** One full bark-texture repeat per this many units of the tree's
 *  normalised space — picks the texel size on trunk AND branches. */
const UV_UNITS_PER_REPEAT = 0.75;

/** Fallback dry-wood tint if the bark texture fails to load. */
const FALLBACK_COLOR = 0x8f6b46;

/* --------------------------- small helpers --------------------------- */

/** Deterministic 0..1 hash from two integer coords + a seed salt. */
function hash2(x: number, z: number, salt: number): number {
  let h = (Math.imul(x, 374761393) + Math.imul(z, 668265263) + Math.imul(salt, 2246822519)) | 0;
  h ^= h >>> 13;
  h = Math.imul(h, 1274126177) | 0;
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

/** Positive modulo for integers. */
function wrap(v: number, m: number): number {
  return ((v % m) + m) % m;
}

/** Box-projection UVs for one baked (normalised) tree geometry: every
 *  vertex projects along its DOMINANT normal axis —
 *    |ny| wins → planar from above (u = x, v = z)
 *    |nx| wins → side wall      (u = z, v = y)
 *    else      → side wall      (u = x, v = y)
 *  scaled so one texture repeat spans UV_UNITS_PER_REPEAT units. The
 *  pixel bark then wraps trunks, forks and branches at an even texel
 *  density (the GLB's own atlas UVs would smear it). */
function addBoxUVs(g: THREE.BufferGeometry): void {
  const pos = g.getAttribute('position');
  const nor = g.getAttribute('normal');
  if (!pos || !nor) return;
  const uv = new Float32Array(pos.count * 2);
  const k = 1 / UV_UNITS_PER_REPEAT;
  for (let i = 0; i < pos.count; i++) {
    const nx = Math.abs(nor.getX(i));
    const ny = Math.abs(nor.getY(i));
    const nz = Math.abs(nor.getZ(i));
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    if (ny >= nx && ny >= nz) {
      uv[i * 2] = x * k;
      uv[i * 2 + 1] = z * k;
    } else if (nx >= nz) {
      uv[i * 2] = z * k;
      uv[i * 2 + 1] = y * k;
    } else {
      uv[i * 2] = x * k;
      uv[i * 2 + 1] = y * k;
    }
  }
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
}

/** Max ground-level radius of a NORMALISED variant geometry: the root
 *  flare band's real silhouette becomes the per-instance collision disc
 *  (scaled by the instance height at placement time). */
function bakeUnitRadius(g: THREE.BufferGeometry): number {
  const pos = g.getAttribute('position');
  let r = 0.02;
  if (!pos) return r;
  for (let vi = 0; vi < pos.count; vi++) {
    if (pos.getY(vi) > BASE_BAND) continue;
    const vx = pos.getX(vi);
    const vz = pos.getZ(vi);
    const d = Math.sqrt(vx * vx + vz * vz);
    if (d > r) r = d;
  }
  return r;
}

export function createDeadTrees(
  scene: THREE.Scene,
  options: DeadTreeSystemOptions
): DeadTreeSystemHandle {
  const block = options.block;
  const gridOffset = options.gridOffset;
  const lowSpec = options.lowSpec ?? false;
  const clearX = options.clearX ?? 0;
  const clearZ = options.clearZ ?? 0;
  const clearR2 = (options.clearRadius ?? 0) ** 2;

  const R = lowSpec ? RADIUS_LOW : RADIUS_DESKTOP;
  const W = 2 * R + 1; // window edge, in cells
  const count = Math.round(W * W * (lowSpec ? DENSITY_LOW : DENSITY) * 1.15); // +15% reject margin
  const VARIANTS = 4; // the four GLB trees

  /* --- fixed per-instance cell residues inside the window (unique, so
   *     two trees never land on the exact same cell and overlap) --- */
  const baseX = new Int32Array(count);
  const baseZ = new Int32Array(count);
  {
    const used = new Set<number>();
    let filled = 0;
    for (let attempt = 0; filled < count && attempt < count * 40; attempt++) {
      const bx = wrap(Math.floor(hash2(filled, 11, 1301 + attempt * 7) * W * 4096), W);
      const bz = wrap(Math.floor(hash2(filled, 17, 1367 + attempt * 11) * W * 4096), W);
      const key = bz * W + bx;
      if (used.has(key)) continue;
      used.add(key);
      baseX[filled] = bx;
      baseZ[filled] = bz;
      filled++;
    }
  }

  /* --- per-instance placement state (read back by pushOut) --- */
  /** World position of each tree's base; unplaced park at y = -1e5. */
  const treeX = new Float32Array(count);
  const treeY = new Float32Array(count).fill(-1e5);
  const treeZ = new Float32Array(count);
  /** Collision radius per placed tree (variant base radius × scale). */
  const treeR = new Float32Array(count);
  /** Which of the five tree variants this instance wears. */
  const treeVariant = new Uint8Array(count);
  const placed = new Uint8Array(count);
  let placedCount = 0;

  let loaded = false;
  /** One InstancedMesh per variant (capacity `count`; foreign instances
   *  park at a zero-scale matrix). */
  const meshes: THREE.InstancedMesh[] = [];
  const geos: THREE.BufferGeometry[] = [];
  const mats: THREE.MeshLambertMaterial[] = [];
  const textures: THREE.Texture[] = [];
  /** Unit-space collision radius per variant (baked at load). */
  const variantRadius = new Float32Array(VARIANTS).fill(0.05);

  const dummy = new THREE.Object3D();
  const zeroMat = new THREE.Matrix4().makeScale(0, 0, 0);
  const slideOut = { x: 0, z: 0 }; // shared pushOut result — zero alloc

  /** Load the GLB, split its four meshes into separate normalised tree
   *  variants and build the InstancedMesh set. */
  function attach(
    gltf: { scene: THREE.Group },
    barkTex: THREE.Texture | null
  ): void {
    const root = gltf.scene;
    root.updateMatrixWorld(true);

    // Each mesh in the pack is ONE complete tree — bake it into world
    // space, then normalise it on its OWN bounds (centre x/z, base on
    // y=0, unit height) so per-instance matrices set the real size.
    const variants: THREE.BufferGeometry[] = [];
    root.traverse((obj) => {
      const meshChild = obj as THREE.Mesh;
      if (!(meshChild as THREE.Mesh).isMesh) return;
      const g = meshChild.geometry.clone();
      for (const name of Object.keys(g.attributes)) {
        if (name !== 'position' && name !== 'normal' && name !== 'uv') {
          g.deleteAttribute(name);
        }
      }
      g.applyMatrix4(meshChild.matrixWorld);
      g.computeBoundingBox();
      const bb = g.boundingBox!;
      const height = Math.max(1e-6, bb.max.y - bb.min.y);
      const cx = (bb.min.x + bb.max.x) / 2;
      const cz = (bb.min.z + bb.max.z) / 2;
      g.translate(-cx, -bb.min.y, -cz);
      g.scale(1 / height, 1 / height, 1 / height);

      // Pixel bark over every trunk and branch (replaces the GLB's
      // atlas UVs, which have no texture behind them anyway).
      addBoxUVs(g);

      // Collision radius: the max ground-level radius of the root-flare
      // band — the disc the player actually slides around.
      variantRadius[variants.length % VARIANTS] = bakeUnitRadius(g);
      variants.push(g);
    });

    // One Lambert material shared by every variant: the same pixel bark
    // as the stumps, shaded exactly like the voxel terrain.
    const material = new THREE.MeshLambertMaterial({
      map: barkTex,
      color: barkTex ? 0xffffff : FALLBACK_COLOR,
      side: THREE.DoubleSide, // dead wood often has open shells
    });

    for (let v = 0; v < Math.min(variants.length, VARIANTS); v++) {
      const mesh = new THREE.InstancedMesh(variants[v], material, count);
      mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
      mesh.frustumCulled = false; // instances span the whole view
      mesh.castShadow = !lowSpec;
      mesh.receiveShadow = true;
      mesh.renderOrder = 1; // above the terrain, under transparent sky
      // park every instance out of sight until the first placement pass
      for (let i = 0; i < count; i++) mesh.setMatrixAt(i, zeroMat);
      scene.add(mesh);
      meshes.push(mesh);
      geos.push(variants[v]);
    }
    mats.push(material);

    loaded = true;
    lastCx = Number.NaN; // force a placement pass on the next update
    lastCz = Number.NaN;
  }

  /* ------------------- placement (world-anchored) ------------------- */

  function placeAll(pcx: number, pcz: number): void {
    if (!loaded || meshes.length === 0) return;
    placedCount = 0;

    for (let i = 0; i < count; i++) {
      // wrap the fixed residue into the window around the player's cell
      const gx = pcx + wrap(baseX[i] - pcx, W) - R;
      const gz = pcz + wrap(baseZ[i] - pcz, W) - R;

      // world position: block centre + per-cell jitter (kept inside the
      // block so the tree stands on its own column)
      const wx = (gx - gridOffset) * block + (hash2(gx, gz, 977) - 0.5) * block * JITTER_FRACTION;
      const wz = (gz - gridOffset) * block + (hash2(gx, gz, 1171) - 0.5) * block * JITTER_FRACTION;

      // keep the lobby spawn showcase clear
      const dx = wx - clearX;
      const dz = wz - clearZ;
      if (dx * dx + dz * dz < clearR2) {
        park(i);
        continue;
      }

      // skip cliff cells: the voxel height across this block must stay
      // within a couple of steps
      const h00 = options.blockHeight(gx, gz);
      const h10 = options.blockHeight(gx + 1, gz);
      const h01 = options.blockHeight(gx, gz + 1);
      const h11 = options.blockHeight(gx + 1, gz + 1);
      const hMax = Math.max(h00, h10, h01, h11);
      const hMin = Math.min(h00, h10, h01, h11);
      if (hMax - hMin > SLOPE_MAX_BLOCKS) {
        park(i);
        continue;
      }

      // variant: a per-cell hash picks which of the four GLB trees
      // grows here (world-anchored — the same spot always regrows the
      // same species, evenly spread across the pack).
      const vHash = hash2(gx, gz, 1307);
      const variant = Math.min(meshes.length - 1, (vHash * meshes.length) | 0);
      treeVariant[i] = variant;

      // height: base range jittered per cell, then the variant's own
      // multiplier keeps the pack's size ranking intact
      const mult = VARIANT_HEIGHT_MULT[variant % VARIANT_HEIGHT_MULT.length];
      const h = (HEIGHT_MIN + HEIGHT_VAR * hash2(gx, gz, 1361)) * mult;
      const sink = Math.max(SINK_MIN, h * SINK_FRACTION);
      const groundY = options.surfaceYAt(wx, wz) - sink;
      const rotY = hash2(gx, gz, 1423) * Math.PI * 2;

      // dead trees lean on a random azimuth (±TILT_MAX rad)
      const tiltAmp = (hash2(gx, gz, 1481) - 0.5) * 2 * TILT_MAX;
      const tiltDir = hash2(gx, gz, 1523) * Math.PI * 2;

      dummy.position.set(wx, groundY, wz);
      dummy.rotation.set(
        Math.sin(tiltDir) * tiltAmp,
        rotY,
        Math.cos(tiltDir) * tiltAmp
      );
      dummy.scale.set(h, h, h);
      dummy.updateMatrix();
      // only the chosen variant's mesh wears this matrix; the others
      // park their instance i at a zero scale
      for (let v = 0; v < meshes.length; v++) {
        meshes[v].setMatrixAt(i, v === variant ? dummy.matrix : zeroMat);
      }

      treeX[i] = wx;
      treeY[i] = groundY;
      treeZ[i] = wz;
      treeR[i] = variantRadius[variant] * h;
      placed[i] = 1;
      placedCount++;
    }

    for (const m of meshes) m.instanceMatrix.needsUpdate = true;
  }

  /** Park an instance out of sight + out of collision. */
  function park(i: number): void {
    for (const m of meshes) m.setMatrixAt(i, zeroMat);
    treeY[i] = -1e5;
    placed[i] = 0;
  }

  let lastCx = Number.NaN;
  let lastCz = Number.NaN;

  /* ------------------------ per-frame driver ------------------------ */

  function update(playerX: number, playerZ: number): void {
    // grid coordinates — same space as terrainChunks.blockHeight()
    const pcx = Math.round(playerX / block) + gridOffset;
    const pcz = Math.round(playerZ / block) + gridOffset;
    if (pcx === lastCx && pcz === lastCz) return;
    lastCx = pcx;
    lastCz = pcz;
    placeAll(pcx, pcz);
  }

  function pushOut(x: number, z: number, radius: number): { x: number; z: number } {
    if (!loaded) {
      slideOut.x = x;
      slideOut.z = z;
      return slideOut;
    }
    for (let i = 0; i < count; i++) {
      if (!placed[i]) continue;
      const dx = x - treeX[i];
      const dz = z - treeZ[i];
      const minD = radius + treeR[i];
      const d2 = dx * dx + dz * dz;
      if (d2 >= minD * minD) continue;
      const d = Math.sqrt(d2);
      if (d > 1e-6) {
        // slide the mover out along the radial — a natural wall-slide
        // around the trunk's baked base silhouette
        const push = minD - d;
        x += (dx / d) * push;
        z += (dz / d) * push;
      } else {
        // dead centre: eject along +X (arbitrary but stable)
        x = treeX[i] + minD;
      }
    }
    slideOut.x = x;
    slideOut.z = z;
    return slideOut;
  }

  function stats(): { instances: number; placed: number } {
    return { instances: count, placed: placedCount };
  }

  function dispose(): void {
    for (const m of meshes) {
      scene.remove(m);
      m.dispose();
    }
    for (const g of geos) g.dispose();
    for (const mat of mats) mat.dispose();
    for (const t of textures) t.dispose();
    meshes.length = 0;
    geos.length = 0;
    mats.length = 0;
    textures.length = 0;
    loaded = false;
  }

  /* ----------------------------- loading ----------------------------- */

  /** Load the pixel bark; resolves null on failure so the trees fall
   *  back to a flat dry-wood tint instead of losing the model. */
  function loadBark(url: string): Promise<THREE.Texture | null> {
    return new Promise((resolve) => {
      new THREE.TextureLoader().load(
        url,
        (t) => {
          t.colorSpace = THREE.SRGBColorSpace;
          t.wrapS = t.wrapT = THREE.RepeatWrapping;
          t.anisotropy = 4;
          textures.push(t);
          resolve(t);
        },
        undefined,
        () => resolve(null)
      );
    });
  }

  new GLTFLoader().load(
    DEAD_TREES_URL,
    (gltf) => {
      loadBark(DEAD_BARK_URL).then((barkTex) => attach(gltf, barkTex));
    },
    undefined,
    (err: unknown) => {
      // load error: the dead trees just don't appear; the game keeps
      // running — but log the REAL reason so broken assets are findable
      const e = err as { message?: string; response?: { status?: number; url?: string } };
      console.warn(
        '[deadTrees] failed to load',
        DEAD_TREES_URL,
        e?.message ?? err,
        e?.response?.status,
        e?.response?.url
      );
    }
  );

  return {
    ready: () => loaded,
    update,
    pushOut,
    stats,
    dispose,
  };
}
