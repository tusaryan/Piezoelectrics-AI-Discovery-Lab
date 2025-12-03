import React, { useState, useEffect } from 'react';
import { AppBar, Toolbar, Typography, Button, Container, Box, CssBaseline, IconButton, Menu, MenuItem, useScrollTrigger, Slide } from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import { Link as ScrollLink } from 'react-scroll';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { motion, AnimatePresence } from 'framer-motion';
import Footer from './Footer';

// Modern Light Theme
const theme = createTheme({
    palette: {
        mode: 'light',
        primary: {
            main: '#0288d1', // Modern Blue
        },
        secondary: {
            main: '#e91e63', // Pink accent
        },
        background: {
            default: '#f5f7fa', // Very light grey/blue
            paper: '#ffffff',
        },
        text: {
            primary: '#263238',
            secondary: '#546e7a',
        },
    },
    typography: {
        fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
        h1: { fontWeight: 700, letterSpacing: '-0.02em' },
        h2: { fontWeight: 600, letterSpacing: '-0.01em' },
        h3: { fontWeight: 600 },
        h4: { fontWeight: 600 },
        h5: { fontWeight: 500 },
        h6: { fontWeight: 500 },
        button: { textTransform: 'none', fontWeight: 600 },
    },
    shape: {
        borderRadius: 12,
    },
    components: {
        MuiButton: {
            styleOverrides: {
                root: {
                    borderRadius: 24, // Pill shaped buttons
                    boxShadow: 'none',
                    '&:hover': {
                        boxShadow: '0px 4px 12px rgba(0,0,0,0.1)',
                    },
                },
            },
        },
        MuiPaper: {
            styleOverrides: {
                root: {
                    boxShadow: '0px 4px 20px rgba(0,0,0,0.05)',
                    border: '1px solid rgba(0,0,0,0.05)',
                },
            },
        },
    },
});

function HideOnScroll(props) {
    const { children, window } = props;
    const trigger = useScrollTrigger({
        target: window ? window() : undefined,
    });

    return (
        <Slide appear={false} direction="down" in={!trigger}>
            {children}
        </Slide>
    );
}

const Layout = ({ children }) => {
    const [anchorElNav, setAnchorElNav] = useState(null);
    const [scrolled, setScrolled] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            setScrolled(window.scrollY > 50);
        };
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const handleOpenNavMenu = (event) => {
        setAnchorElNav(event.currentTarget);
    };

    const handleCloseNavMenu = () => {
        setAnchorElNav(null);
    };

    const navItems = [
        { label: 'Home', to: 'home' },
        { label: 'Predict', to: 'predict' },
        { label: 'Insights', to: 'insights' },
        { label: 'Retrain', to: 'train' },
        { label: 'Dataset', to: 'dataset' },
        { label: 'About', to: 'about' },
    ];

    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <HideOnScroll>
                <AppBar
                    position="fixed"
                    color="transparent"
                    elevation={scrolled ? 4 : 0}
                    sx={{
                        backdropFilter: 'blur(20px)',
                        backgroundColor: scrolled ? 'rgba(255, 255, 255, 0.8)' : 'transparent',
                        transition: 'all 0.3s ease',
                        borderBottom: scrolled ? '1px solid rgba(0,0,0,0.05)' : 'none',
                    }}
                >
                    <Container maxWidth="xl">
                        <Toolbar disableGutters sx={{ height: 80 }}>
                            {/* Desktop Logo */}
                            <Typography
                                variant="h5"
                                noWrap
                                component={ScrollLink}
                                to="home"
                                smooth={true}
                                duration={500}
                                sx={{
                                    mr: 2,
                                    display: { xs: 'none', md: 'flex' },
                                    fontFamily: 'Inter',
                                    fontWeight: 800,
                                    letterSpacing: '-0.05em',
                                    color: 'primary.main',
                                    textDecoration: 'none',
                                    cursor: 'pointer',
                                    fontSize: '1.8rem'
                                }}
                            >
                                PIEZO.AI
                            </Typography>

                            {/* Mobile Menu */}
                            <Box sx={{ flexGrow: 1, display: { xs: 'flex', md: 'none' } }}>
                                <IconButton
                                    size="large"
                                    onClick={handleOpenNavMenu}
                                    color="primary"
                                >
                                    <MenuIcon />
                                </IconButton>
                                <Menu
                                    id="menu-appbar"
                                    anchorEl={anchorElNav}
                                    anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
                                    keepMounted
                                    transformOrigin={{ vertical: 'top', horizontal: 'left' }}
                                    open={Boolean(anchorElNav)}
                                    onClose={handleCloseNavMenu}
                                    sx={{ display: { xs: 'block', md: 'none' } }}
                                >
                                    {navItems.map((item) => (
                                        <MenuItem key={item.label} onClick={handleCloseNavMenu}>
                                            <ScrollLink to={item.to} smooth={true} duration={500} onClick={handleCloseNavMenu}>
                                                <Typography textAlign="center">{item.label}</Typography>
                                            </ScrollLink>
                                        </MenuItem>
                                    ))}
                                </Menu>
                            </Box>

                            {/* Mobile Logo */}
                            <Typography
                                variant="h5"
                                noWrap
                                component={ScrollLink}
                                to="home"
                                smooth={true}
                                duration={500}
                                sx={{
                                    mr: 2,
                                    display: { xs: 'flex', md: 'none' },
                                    flexGrow: 1,
                                    fontFamily: 'Inter',
                                    fontWeight: 800,
                                    color: 'primary.main',
                                    textDecoration: 'none',
                                    cursor: 'pointer'
                                }}
                            >
                                PIEZO.AI
                            </Typography>

                            {/* Desktop Menu */}
                            <Box sx={{ flexGrow: 1, display: { xs: 'none', md: 'flex' }, justifyContent: 'flex-end', gap: 2 }}>
                                {navItems.map((item) => (
                                    <ScrollLink
                                        key={item.label}
                                        to={item.to}
                                        smooth={true}
                                        duration={500}
                                        spy={true}
                                        exact="true"
                                        offset={-80}
                                    >
                                        <Button
                                            sx={{
                                                color: 'text.primary',
                                                display: 'block',
                                                fontSize: '1rem',
                                                '&:hover': { color: 'primary.main', bgcolor: 'transparent' }
                                            }}
                                        >
                                            {item.label}
                                        </Button>
                                    </ScrollLink>
                                ))}
                            </Box>
                        </Toolbar>
                    </Container>
                </AppBar>
            </HideOnScroll>

            <Box component="main" sx={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
                {children}
            </Box>
            <Footer />
        </ThemeProvider>
    );
};

export default Layout;
