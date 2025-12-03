import React, { useState, useEffect } from 'react';
import { Box, Typography, Paper, Grid, CircularProgress, Alert } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import axios from 'axios';

const ModelInsights = ({ refreshTrigger, isTraining, isTrained }) => {
    const [comparisonD33, setComparisonD33] = useState([]);
    const [comparisonTc, setComparisonTc] = useState([]);
    const [scatterD33, setScatterD33] = useState([]);
    const [scatterTc, setScatterTc] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            // If not trained and not training, we might not have data, but let's try fetching anyway
            // or just skip if we know it's not trained. 
            // However, on first load, isTrained might be false until status returns.
            // Let's fetch and handle empty/error.

            try {
                const response = await axios.get('http://localhost:8000/insights');
                setComparisonD33(response.data.comparison_d33 || []);
                setComparisonTc(response.data.comparison_tc || []);
                setScatterD33(response.data.scatter_d33 || []);
                setScatterTc(response.data.scatter_tc || []);
            } catch (error) {
                console.error("Error fetching insights:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [refreshTrigger]);

    if (isTraining) {
        return (
            <Box sx={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
                <CircularProgress size={60} sx={{ mb: 2 }} />
                <Typography variant="h6" color="text.secondary">Training in progress... Insights will update shortly.</Typography>
            </Box>
        );
    }

    if (!isTrained && !loading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
                <Alert severity="info" variant="outlined" sx={{ fontSize: '1.2rem', px: 4, py: 2 }}>
                    Models have not been trained yet. Please go to the Retraining section to generate insights.
                </Alert>
            </Box>
        );
    }

    if (loading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
                <CircularProgress size={60} />
            </Box>
        );
    }

    return (
        <Box sx={{ width: '100%', maxWidth: 1200, mx: 'auto' }}>
            <Box sx={{ textAlign: 'center', mb: 6 }}>
                <Typography variant="h3" gutterBottom color="primary">Model Insights</Typography>
                <Typography variant="body1" color="text.secondary">
                    Analyze the performance of our machine learning models.
                </Typography>
            </Box>

            <Grid container spacing={6}>
                {/* Comparison d33 */}
                <Grid size={{ xs: 12 }}>
                    <Paper sx={{ p: 4, borderRadius: 4, boxShadow: '0 4px 20px rgba(0,0,0,0.05)' }}>
                        <Typography variant="h5" gutterBottom fontWeight="bold" color="text.primary">Algorithm Comparison (d<sub>33</sub>)</Typography>
                        {comparisonD33 && comparisonD33.length > 0 ? (
                            <ResponsiveContainer width="100%" height={500}>
                                <BarChart data={comparisonD33} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
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
                </Grid>

                {/* Comparison Tc */}
                <Grid size={{ xs: 12 }}>
                    <Paper sx={{ p: 4, borderRadius: 4, boxShadow: '0 4px 20px rgba(0,0,0,0.05)' }}>
                        <Typography variant="h5" gutterBottom fontWeight="bold" color="text.primary">Algorithm Comparison (T<sub>c</sub>)</Typography>
                        {comparisonTc && comparisonTc.length > 0 ? (
                            <ResponsiveContainer width="100%" height={500}>
                                <BarChart data={comparisonTc} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
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
                </Grid>

                <Grid size={{ xs: 12 }}>
                    <Paper sx={{ p: 4, borderRadius: 4, boxShadow: '0 4px 20px rgba(0,0,0,0.05)' }}>
                        <Typography variant="h5" gutterBottom fontWeight="bold" color="text.primary">Predicted vs Actual (d<sub>33</sub>)</Typography>
                        {scatterD33 && scatterD33.length > 0 ? (
                            <ResponsiveContainer width="100%" height={600}>
                                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                    <CartesianGrid stroke="#eee" />
                                    <XAxis type="number" dataKey="x" name="Actual" unit=" pC/N" stroke="#546e7a" label={{ value: 'Actual d33 (pC/N)', position: 'bottom', offset: 0 }} />
                                    <YAxis type="number" dataKey="y" name="Predicted" unit=" pC/N" stroke="#546e7a" label={{ value: 'Predicted d33 (pC/N)', angle: -90, position: 'insideLeft' }} />
                                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ borderRadius: 8, border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                                    <Scatter name="d33" data={scatterD33} fill="#0288d1" />
                                </ScatterChart>
                            </ResponsiveContainer>
                        ) : (
                            <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                <Typography color="text.secondary">No scatter data available.</Typography>
                            </Box>
                        )}
                    </Paper>
                </Grid>

                <Grid size={{ xs: 12 }}>
                    <Paper sx={{ p: 4, borderRadius: 4, boxShadow: '0 4px 20px rgba(0,0,0,0.05)' }}>
                        <Typography variant="h5" gutterBottom fontWeight="bold" color="text.primary">Predicted vs Actual (T<sub>c</sub>)</Typography>
                        {scatterTc && scatterTc.length > 0 ? (
                            <ResponsiveContainer width="100%" height={600}>
                                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                    <CartesianGrid stroke="#eee" />
                                    <XAxis type="number" dataKey="x" name="Actual" unit=" °C" stroke="#546e7a" label={{ value: 'Actual Tc (°C)', position: 'bottom', offset: 0 }} />
                                    <YAxis type="number" dataKey="y" name="Predicted" unit=" °C" stroke="#546e7a" label={{ value: 'Predicted Tc (°C)', angle: -90, position: 'insideLeft' }} />
                                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ borderRadius: 8, border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                                    <Scatter name="Tc" data={scatterTc} fill="#e91e63" />
                                </ScatterChart>
                            </ResponsiveContainer>
                        ) : (
                            <Box sx={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                <Typography color="text.secondary">No scatter data available.</Typography>
                            </Box>
                        )}
                    </Paper>
                </Grid>
            </Grid>
        </Box>
    );
};

export default ModelInsights;
