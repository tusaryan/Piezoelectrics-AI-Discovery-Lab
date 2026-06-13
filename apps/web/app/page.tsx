/**
 * Piezo.AI — Landing / Redirect Page
 * Redirects to /dashboard on load.
 */

import { redirect } from "next/navigation";

export default function Home() {
  redirect("/dashboard");
}
