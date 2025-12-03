import React, { useState, useEffect } from 'react';
import { Box, Typography, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TablePagination, CircularProgress, Chip } from '@mui/material';
import axios from 'axios';

const DatasetViewer = () => {
    const [page, setPage] = useState(0);
    const [rowsPerPage, setRowsPerPage] = useState(10);
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
                const response = await axios.get(`${apiUrl}/dataset`);
                setData(response.data);
                setLoading(false);
            } catch (error) {
                console.error("Error fetching dataset:", error);
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    const handleChangePage = (event, newPage) => {
        setPage(newPage);
    };

    const handleChangeRowsPerPage = (event) => {
        setRowsPerPage(+event.target.value);
        setPage(0);
    };

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
                <Typography variant="h3" gutterBottom color="primary">Dataset Explorer</Typography>
                <Typography variant="body1" color="text.secondary">
                    Browse the chemical compositions and properties used to train our models.
                </Typography>
            </Box>

            <Paper sx={{ width: '100%', overflow: 'hidden', borderRadius: 4, boxShadow: '0 4px 20px rgba(0,0,0,0.05)' }}>
                <TableContainer sx={{ maxHeight: 600 }}>
                    <Table stickyHeader aria-label="sticky table">
                        <TableHead>
                            <TableRow>
                                <TableCell sx={{ fontWeight: 'bold', bgcolor: 'background.default' }}>ID</TableCell>
                                <TableCell sx={{ fontWeight: 'bold', bgcolor: 'background.default' }}>Chemical Formula</TableCell>
                                <TableCell align="right" sx={{ fontWeight: 'bold', bgcolor: 'background.default' }}>d<sub>33</sub> (pC/N)</TableCell>
                                <TableCell align="right" sx={{ fontWeight: 'bold', bgcolor: 'background.default' }}>T<sub>c</sub> (°C)</TableCell>
                                <TableCell align="center" sx={{ fontWeight: 'bold', bgcolor: 'background.default' }}>Source</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {data
                                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                                .map((row) => (
                                    <TableRow hover role="checkbox" tabIndex={-1} key={row.id}>
                                        <TableCell>{row.id}</TableCell>
                                        <TableCell sx={{ fontFamily: 'monospace', fontWeight: 500 }}>{row.formula}</TableCell>
                                        <TableCell align="right">{row.d33}</TableCell>
                                        <TableCell align="right">{row.tc}</TableCell>
                                        <TableCell align="center">
                                            <Chip label={row.source} size="small" color="primary" variant="outlined" />
                                        </TableCell>
                                    </TableRow>
                                ))}
                        </TableBody>
                    </Table>
                </TableContainer>
                <TablePagination
                    rowsPerPageOptions={[10, 25, 100]}
                    component="div"
                    count={data.length}
                    rowsPerPage={rowsPerPage}
                    page={page}
                    onPageChange={handleChangePage}
                    onRowsPerPageChange={handleChangeRowsPerPage}
                />
            </Paper>
        </Box>
    );
};

export default DatasetViewer;
