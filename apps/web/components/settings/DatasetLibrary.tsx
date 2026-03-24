"use client";

import { useEffect, useState } from "react";
import { Trash2, ExternalLink, Loader2, Database } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";

interface Dataset {
  id: string;
  name: string;
  status: string;
  row_count: number;
  has_d33: boolean;
  has_tc: boolean;
  created_at: string;
}

export function DatasetLibrary() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  const fetchDatasets = async () => {
    try {
      const res = await fetch("/api/v1/datasets");
      const json = await res.json();
      setDatasets(json.data || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  const handleDelete = async (id: string) => {
    if (!confirm("Are you sure you want to delete this dataset?")) return;
    try {
      await fetch(`/api/v1/datasets/${id}`, { method: "DELETE" });
      fetchDatasets();
    } catch (e) {
      console.error(e);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center p-8">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (datasets.length === 0) {
    return (
      <div className="text-center p-8 text-muted-foreground">
        <Database className="w-8 h-8 mx-auto mb-3 opacity-20" />
        <p>No datasets found.</p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto hide-scrollbar p-2">
      <table className="w-full text-sm text-left">
        <thead className="text-xs text-muted-foreground bg-muted/50">
          <tr>
            <th className="px-4 py-3 font-medium rounded-tl-md">Dataset Name</th>
            <th className="px-4 py-3 font-medium">Rows</th>
            <th className="px-4 py-3 font-medium">Status</th>
            <th className="px-4 py-3 font-medium rounded-tr-md text-right">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {datasets.map((ds) => (
            <tr key={ds.id} className="hover:bg-muted/20 transition-colors">
              <td className="px-4 py-3 font-medium truncate max-w-[150px]" title={ds.name}>{ds.name}</td>
              <td className="px-4 py-3">{ds.row_count}</td>
              <td className="px-4 py-3">
                <span className={`px-2 py-1 rounded-full text-xs ${
                  ds.status === 'ready' ? 'bg-emerald-500/10 text-emerald-500' : 'bg-yellow-500/10 text-yellow-500'
                }`}>
                  {ds.status}
                </span>
              </td>
              <td className="px-4 py-3 text-right">
                <div className="flex justify-end gap-2">
                  <Button variant="ghost" size="icon" onClick={() => router.push(`/dataset?id=${ds.id}&tab=table`)} className="h-8 w-8 text-blue-500 hover:text-blue-600 hover:bg-blue-500/10" title="Manage Dataset">
                    <ExternalLink className="w-4 h-4" />
                  </Button>
                  <Button variant="ghost" size="icon" onClick={() => handleDelete(ds.id)} className="h-8 w-8 text-red-500 hover:text-red-600 hover:bg-red-500/10" title="Delete Dataset">
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
