"use client";

import { useState, useMemo, useRef, useCallback } from "react";
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  ColumnDef,
  getSortedRowModel,
  SortingState,
  getFilteredRowModel,
  ColumnResizeMode,
} from "@tanstack/react-table";
import { useVirtualizer } from "@tanstack/react-virtual";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, Search } from "lucide-react";

interface Material {
  id: string;
  dataset_id: string;
  formula: string;
  sintering_temp: number | null;
  d33: number | null;
  tc: number | null;
  is_imputed: boolean;
  is_tc_ai_generated: boolean;
}

interface MaterialsTableProps {
  materials: Material[];
}

export function MaterialsTable({ materials }: MaterialsTableProps) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [globalFilter, setGlobalFilter] = useState('');
  const [columnResizeMode] = useState<ColumnResizeMode>('onChange');
  const tableContainerRef = useRef<HTMLDivElement>(null);

  const columns = useMemo<ColumnDef<Material>[]>(
    () => [
      {
        accessorKey: "formula",
        header: "Formula",
        size: 350,
        minSize: 150,
        maxSize: 600,
        cell: (info) => {
          const val = info.getValue<string>();
          return (
            <div className="font-medium text-primary truncate" title={val}>
              {val}
            </div>
          );
        },
      },
      {
        accessorKey: "sintering_temp",
        header: "Sintering Temp (°C)",
        size: 160,
        minSize: 100,
        maxSize: 300,
        cell: (info) => {
          const val = info.getValue<number | null>();
          const display = val !== null && val !== undefined ? val.toFixed(1) : "-";
          return <span title={display}>{display}</span>;
        },
      },
      {
        accessorKey: "d33",
        header: "d33 (pC/N)",
        size: 150,
        minSize: 100,
        maxSize: 300,
        cell: (info) => {
          const val = info.getValue<number | null>();
          const display = val !== null && val !== undefined ? val.toFixed(2) : "-";
          return <span title={display}>{display}</span>;
        },
      },
      {
        accessorKey: "tc",
        header: "Tc (°C)",
        size: 150,
        minSize: 100,
        maxSize: 300,
        cell: (info) => {
          const val = info.getValue<number | null>();
          const imputed = info.row.original.is_tc_ai_generated;
          if (val === null || val === undefined) return <span title="-">-</span>;
          const display = val.toFixed(2);
          return (
            <span
              title={imputed ? `${display} (AI-generated)` : display}
              className={imputed ? "text-yellow-600 font-semibold px-2 py-0.5 bg-yellow-500/10 rounded" : ""}
            >
              {display}
            </span>
          );
        },
      },
    ],
    []
  );

  const table = useReactTable({
    data: materials,
    columns,
    columnResizeMode,
    state: {
      sorting,
      globalFilter,
    },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  });

  const { rows } = table.getRowModel();

  const rowVirtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => tableContainerRef.current,
    estimateSize: () => 48,
    overscan: 10,
  });

  // Get column sizes from table state for consistent widths
  const columnSizing = table.getState().columnSizing;
  const getColWidth = useCallback((colId: string) => {
    const col = table.getColumn(colId);
    if (!col) return 150;
    return columnSizing[colId] ?? col.getSize();
  }, [table, columnSizing]);

  const totalWidth = table.getHeaderGroups()[0]?.headers.reduce(
    (sum, h) => sum + getColWidth(h.id), 0
  ) ?? 810;

  return (
    <div className="w-full mx-auto space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-semibold mb-1">Dataset Explorer</h3>
          <p className="text-sm text-muted-foreground">Virtualised view of the processed materials. Drag column edges to resize.</p>
        </div>
        <div className="flex px-4 py-2 bg-muted/50 border rounded-lg gap-2 items-center">
          <Search className="w-4 h-4 text-muted-foreground" />
          <input 
            type="text" 
            placeholder="Search formula..." 
            className="bg-transparent border-none outline-none text-sm w-48"
            value={globalFilter ?? ''}
            onChange={(e) => setGlobalFilter(String(e.target.value))}
          />
        </div>
      </div>

      <div className="rounded-xl border bg-card overflow-hidden shadow-sm">
        <div 
          ref={tableContainerRef}
          className="overflow-auto max-h-[600px] w-full"
        >
          {materials.length === 0 ? (
            <div className="h-64 flex flex-col items-center justify-center text-muted-foreground">
              <Loader2 className="w-8 h-8 animate-spin mb-4 text-primary" />
              <p>Loading table data...</p>
            </div>
          ) : (
            <div style={{ minWidth: `${totalWidth}px` }}>
              {/* Fixed header with resize handles */}
              <div className="sticky top-0 bg-card z-10 border-b shadow-[0_1px_0_0_theme(colors.border)]">
                {table.getHeaderGroups().map(headerGroup => (
                  <div key={headerGroup.id} className="flex">
                    {headerGroup.headers.map(header => (
                      <div
                        key={header.id}
                        style={{ width: getColWidth(header.id), position: 'relative' }}
                        className="h-12 px-4 flex items-center font-medium text-muted-foreground cursor-pointer select-none text-sm shrink-0 group"
                        onClick={header.column.getToggleSortingHandler()}
                      >
                        <span className="truncate">
                          {flexRender(header.column.columnDef.header, header.getContext())}
                        </span>
                        {{
                          asc: <span className="text-xs ml-1">↑</span>,
                          desc: <span className="text-xs ml-1">↓</span>,
                        }[header.column.getIsSorted() as string] ?? null}

                        {/* Column resize handle */}
                        <div
                          onMouseDown={header.getResizeHandler()}
                          onTouchStart={header.getResizeHandler()}
                          onClick={(e) => e.stopPropagation()}
                          className={`absolute right-0 top-0 h-full w-1 cursor-col-resize select-none touch-none transition-colors
                            ${header.column.getIsResizing() ? 'bg-primary' : 'bg-transparent group-hover:bg-border hover:!bg-primary/60'}`}
                        />
                      </div>
                    ))}
                  </div>
                ))}
              </div>
              
              {/* Virtualized rows */}
              <div
                style={{
                  height: `${rowVirtualizer.getTotalSize()}px`,
                  position: 'relative',
                }}
              >
                <AnimatePresence>
                  {rowVirtualizer.getVirtualItems().map((virtualRow) => {
                    const row = rows[virtualRow.index];
                    return (
                      <motion.div
                        key={row.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="flex border-b transition-colors hover:bg-muted/50 absolute top-0 left-0"
                        style={{
                          height: `${virtualRow.size}px`,
                          transform: `translateY(${virtualRow.start}px)`,
                          width: `${totalWidth}px`,
                        }}
                      >
                        {row.getVisibleCells().map(cell => (
                          <div 
                            key={cell.id} 
                            style={{ width: getColWidth(cell.column.id) }}
                            className="px-4 flex items-center overflow-hidden shrink-0 truncate"
                          >
                            {flexRender(cell.column.columnDef.cell, cell.getContext())}
                          </div>
                        ))}
                      </motion.div>
                    );
                  })}
                </AnimatePresence>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
