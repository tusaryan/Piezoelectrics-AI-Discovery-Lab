export const features = {
  composite: process.env.NEXT_PUBLIC_ENABLE_COMPOSITE === 'true',
  hardness: process.env.NEXT_PUBLIC_ENABLE_HARDNESS === 'true',
  gnn: process.env.NEXT_PUBLIC_ENABLE_GNN === 'true',
  agent: process.env.NEXT_PUBLIC_ENABLE_AGENT === 'true',
  voice: process.env.NEXT_PUBLIC_ENABLE_VOICE === 'true',
};
