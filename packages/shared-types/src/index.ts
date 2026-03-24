export interface ApiResponse<T = any> {
    success: boolean;
    data: T | null;
    error: ApiError | null;
    meta?: ResponseMeta;
}

export interface ApiError {
    code: string;
    message: string;
    field?: string;
    row?: number;
    suggestions?: string[];
}

export interface ResponseMeta {
    timestamp: string;
    trace_id?: string;
    version?: string;
}
