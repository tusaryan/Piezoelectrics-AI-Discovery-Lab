import { registerOTel } from '@vercel/otel';

export function register() {
  registerOTel({ serviceName: 'piezo-ai-web' });
}
