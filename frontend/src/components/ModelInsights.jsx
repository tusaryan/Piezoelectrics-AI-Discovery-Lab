import React, { useEffect, useState } from 'react';
import { Box, Typography, Paper, Stack, CircularProgress, Alert, Button, Grid } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import RefreshIcon from '@mui/icons-material/Refresh';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import DownloadIcon from '@mui/icons-material/Download';
import axios from 'axios';
import useTrainingStatus from '../hooks/useTrainingStatus';

const ModelInsights = () => {
    const [insights, setInsights] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const { isTraining, isTrained } = useTrainingStatus();

    const fetchInsights = async () => {
        try {
            setLoading(true);
            const response = await axios.get('http://localhost:8000/insights');
            setInsights(response.data);
            setError(null);
        } catch (err) {
            setError("Failed to load model insights. Please ensure models are trained.");
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleExport = async () => {
        try {
            const response = await axios.get('http://localhost:8000/export-report', {
                responseType: 'blob',
            });
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', 'piezo_ai_report.pdf');
            document.body.appendChild(link);
            link.click();
            link.remove();
        } catch (err) {
            console.error("Failed to download report:", err);
            setError("Failed to download report.");
        }
    };

    useEffect(() => {
        if (!isTraining) {
            fetchInsights();
        }
    }, [isTraining]);

    if (isTraining) {
        return (
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '50vh' }}>
                <CircularProgress size={60} />
                <Typography variant="h6" sx={{ mt: 2 }}>Training in progress... Insights will update shortly.</Typography>
            </Box>
        );
    }

    if (loading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
                <CircularProgress />
            </Box>
        );
    }

    if (error) {
        return <Alert severity="warning" sx={{ mt: 4 }}>{error}</Alert>;
    }

    if (!insights || (!insights.comparison_d33 && !insights.comparison_tc && !insights.scatter_d33 && !insights.scatter_tc)) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
                <Alert severity="info" variant="outlined" sx={{ fontSize: '1.2rem', px: 4, py: 2 }}>
                    Models have not been trained yet. Please go to the Retraining section to generate insights.
                </Alert>
            </Box>
        );
    }

    const { comparison_d33, comparison_tc, scatter_d33, scatter_tc } = insights;

    const getBestModel = (data) => {
        if (!data || data.length === 0) return null;
        return data.reduce((prev, current) => (prev.R2 > current.R2) ? prev : current);
    };

    const bestD33 = getBestModel(insights.comparison_d33);
    const bestTc = getBestModel(insights.comparison_tc);

    return (
        <Box sx={{ width: '100%', maxWidth: 1200, mx: 'auto', pb: 8 }}>
            <Box sx={{ textAlign: 'center', mb: 6 }}>
                <Typography variant="h3" gutterBottom color="primary">Model Insights</Typography>
                <Typography variant="body1" color="text.secondary">
                    Detailed performance analysis of trained models.
                </Typography>
            </Box>

            <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 4 }}>
                <Button
                    variant="outlined"
                    startIcon={<DownloadIcon />}
                    onClick={handleExport}
                    sx={{ mr: 2 }}
                >
                    Export Report
                </Button>
                <Button startIcon={<RefreshIcon />} onClick={fetchInsights}>
                    Refresh
                </Button>
            </Box>

            {(bestD33 || bestTc) && (
                <Paper sx={{ p: 3, mb: 6, bgcolor: '#fff3e0', border: '1px solid #ffe0b2', borderRadius: 2 }}>
                    <Typography variant="h6" fontWeight="bold" color="warning.dark" gutterBottom>
                        Recommended Models
                    </Typography>
                    <Typography variant="body2" paragraph>
                        Based on the R² score (Coefficient of Determination), the following models are performing best:
                    </Typography>
                    <Grid container spacing={3}>
                        {bestD33 && (
                            <Grid item xs={12} md={6}>
                                <Paper elevation={0} sx={{ p: 2, bgcolor: 'rgba(255, 255, 255, 0.5)' }}>
                                    <Typography variant="subtitle2" fontWeight="bold">Best for d<sub>33</sub></Typography>
                                    <Typography variant="h5" color="primary.main" gutterBottom>{bestD33.Model}</Typography>
                                    <Typography variant="body2"><strong>R² Score:</strong> {bestD33.R2.toFixed(4)}</Typography>
                                    <Typography variant="body2"><strong>RMSE:</strong> {bestD33.RMSE.toFixed(4)}</Typography>
                                </Paper>
                            </Grid>
                        )}
                        {bestTc && (
                            <Grid item xs={12} md={6}>
                                <Paper elevation={0} sx={{ p: 2, bgcolor: 'rgba(255, 255, 255, 0.5)' }}>
                                    <Typography variant="subtitle2" fontWeight="bold">Best for T<sub>c</sub></Typography>
                                    <Typography variant="h5" color="secondary.main" gutterBottom>{bestTc.Model}</Typography>
                                    <Typography variant="body2"><strong>R² Score:</strong> {bestTc.R2.toFixed(4)}</Typography>
                                    <Typography variant="body2"><strong>RMSE:</strong> {bestTc.RMSE.toFixed(4)}</Typography>
                                </Paper>
                            </Grid>
                        )}
                    </Grid>
                </Paper>
            )}

            <Stack spacing={6} sx={{ width: '100%' }}>
                {/* Comparison d33 */}
                <Paper sx={{ p: 4, borderRadius: 4, boxShadow: '0 4px 20px rgba(0,0,0,0.05)', width: '100%' }}>
                    <Typography variant="h5" gutterBottom fontWeight="bold" color="text.primary">Algorithm Comparison (d<sub>33</sub>)</Typography>
                    {comparison_d33 && comparison_d33.length > 0 ? (
                        <ResponsiveContainer width="100%" height={500}>
                            <BarChart data={comparison_d33} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                                <XAxis dataKey="Model" stroke="#546e7a" tick={{ fontSize: 12 }} />
                                <YAxis yAxisId="left" orientation="left" stroke="#0288d1" label={{ value: 'R² Score', angle: -90, position: 'insideLeft' }} />
                                <YAxis yAxisId="right" orientation="right" stroke="#e91e63" label={{ value: 'RMSE', angle: 90, position: 'insideRight' }} />
                                <Tooltip contentStyle={{ borderRadius: 8, border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                                <Legend />
                                <Bar yAxisId="left" dataKey="R2" fill="#0288d1" name="R² Score" radius={[4, 4, 0, 0]} />
                                <Bar yAxisId="right" dataKey="RMSE" fill="#e91e63" name="RMSE" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    ) : (
                        <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <Typography color="text.secondary">No comparison data available for d33. Train models to see insights.</Typography>
                        </Box>
                    )}
                </Paper>

                {/* Comparison Tc */}
                <Paper sx={{ p: 4, borderRadius: 4, boxShadow: '0 4px 20px rgba(0,0,0,0.05)', width: '100%' }}>
                    <Typography variant="h5" gutterBottom fontWeight="bold" color="text.primary">Algorithm Comparison (T<sub>c</sub>)</Typography>
                    {comparison_tc && comparison_tc.length > 0 ? (
                        <ResponsiveContainer width="100%" height={500}>
                            <BarChart data={comparison_tc} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                                <XAxis dataKey="Model" stroke="#546e7a" tick={{ fontSize: 12 }} />
                                <YAxis yAxisId="left" orientation="left" stroke="#0288d1" label={{ value: 'R² Score', angle: -90, position: 'insideLeft' }} />
                                <YAxis yAxisId="right" orientation="right" stroke="#e91e63" label={{ value: 'RMSE', angle: 90, position: 'insideRight' }} />
                                <Tooltip contentStyle={{ borderRadius: 8, border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                                <Legend />
                                <Bar yAxisId="left" dataKey="R2" fill="#0288d1" name="R² Score" radius={[4, 4, 0, 0]} />
                                <Bar yAxisId="right" dataKey="RMSE" fill="#e91e63" name="RMSE" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    ) : (
                        <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <Typography color="text.secondary">No comparison data available for Tc. Train models to see insights.</Typography>
                        </Box>
                    )}
                </Paper>

                <Paper sx={{ p: 4, borderRadius: 4, boxShadow: '0 4px 20px rgba(0,0,0,0.05)', width: '100%' }}>
                    <Typography variant="h5" gutterBottom fontWeight="bold" color="text.primary">Predicted vs Actual (d<sub>33</sub>)</Typography>
                    {scatter_d33 && scatter_d33.length > 0 ? (
                        <ResponsiveContainer width="100%" height={600}>
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid stroke="#eee" />
                                <XAxis type="number" dataKey="x" name="Actual" unit=" pC/N" stroke="#546e7a" label={{ value: 'Actual d33 (pC/N)', position: 'bottom', offset: 0 }} />
                                <YAxis type="number" dataKey="y" name="Predicted" unit=" pC/N" stroke="#546e7a" label={{ value: 'Predicted d33 (pC/N)', angle: -90, position: 'insideLeft' }} />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ borderRadius: 8, border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                                <Scatter name="d33" data={scatter_d33} fill="#0288d1" />
                            </ScatterChart>
                        </ResponsiveContainer>
                    ) : (
                        <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <Typography color="text.secondary">No scatter data available.</Typography>
                        </Box>
                    )}
                </Paper>

                <Paper sx={{ p: 4, borderRadius: 4, boxShadow: '0 4px 20px rgba(0,0,0,0.05)', width: '100%' }}>
                    <Typography variant="h5" gutterBottom fontWeight="bold" color="text.primary">Predicted vs Actual (T<sub>c</sub>)</Typography>
                    {scatter_tc && scatter_tc.length > 0 ? (
                        <ResponsiveContainer width="100%" height={600}>
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid stroke="#eee" />
                                <XAxis type="number" dataKey="x" name="Actual" unit=" °C" stroke="#546e7a" label={{ value: 'Actual Tc (°C)', position: 'bottom', offset: 0 }} />
                                <YAxis type="number" dataKey="y" name="Predicted" unit=" °C" stroke="#546e7a" label={{ value: 'Predicted Tc (°C)', angle: -90, position: 'insideLeft' }} />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ borderRadius: 8, border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                                <Scatter name="Tc" data={scatter_tc} fill="#e91e63" />
                            </ScatterChart>
                        </ResponsiveContainer>
                    ) : (
                        <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <Typography color="text.secondary">No scatter data available.</Typography>
                        </Box>
                    )}
                </Paper>
            </Stack>
        </Box>
    );
};

export default ModelInsights;
