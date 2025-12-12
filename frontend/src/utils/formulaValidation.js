/**
 * Supported Elements for the Prediction Model
 * This list is modular and can be expanded in the future.
 */
export const SUPPORTED_ELEMENTS = [
    'Ag', 'Al', 'B', 'Ba', 'Bi', 'C', 'Ca', 'Fe', 'Hf', 'Ho', 'K',
    'Li', 'Mn', 'Na', 'Nb', 'O', 'Pr', 'Sb', 'Sc', 'Sr', 'Ta', 'Ti',
    'Zn', 'Zr'
];

/**
 * Validates a chemical formula against supported elements and domain rules.
 * @param {string} formula 
 * @returns {object} { isValid: boolean, errorTitle: string, userMessage: string }
 */
export const validateFormula = (formula) => {
    if (!formula) {
        return {
            isValid: false,
            errorTitle: "Input Error",
            userMessage: "Please enter a valid chemical formula."
        };
    }

    // 0. Sanity Check: Detect Invalid Special Characters
    // Allowed: A-Z, a-z, 0-9, ., (), [], - (for solid solutions)
    const invalidCharRegex = /[^a-zA-Z0-9().[\]-]/;
    if (invalidCharRegex.test(formula)) {
        return {
            isValid: false,
            errorTitle: "Invalid Characters Detected",
            userMessage: "The formula contains special characters that are not recognized in chemical notation. Please use only element symbols, numbers, parentheses, and hyphens."
        };
    }

    // 1. Parsing Check: Extract Elements
    // Matches Capital letter optionally followed by lowercase letter
    const elementRegex = /([A-Z][a-z]?)/g;
    const foundElements = formula.match(elementRegex);

    if (!foundElements || foundElements.length === 0) {
        return {
            isValid: false,
            errorTitle: "Format Error",
            userMessage: "We are currently working on to smoothen our virtual lab experience. Please ensure your formula contains recognized chemical symbols (e.g., Na, Nb)."
        };
    }

    const uniqueElements = new Set(foundElements);

    // 2. Unsupported Element Check
    for (let el of uniqueElements) {
        if (!SUPPORTED_ELEMENTS.includes(el)) {
            return {
                isValid: false,
                errorTitle: `Unsupported Element: ${el}`,
                userMessage: `We are currently expanding our model capability to include more elements like ${el} and enhance prediction accuracy.`
            };
        }
    }

    // 3. Domain Logic: "KN-based" and "KNN-based" Verification
    // Mandatory Elements: Potassium (K), Niobium (Nb), Oxygen (O)
    const hasK = uniqueElements.has('K');
    const hasNb = uniqueElements.has('Nb');
    const hasO = uniqueElements.has('O');
    const hasNa = uniqueElements.has('Na');

    // Identify missing mandatory elements
    const missingElements = [];
    if (!hasK) missingElements.push('Potassium (K)');
    if (!hasNb) missingElements.push('Niobium (Nb)');
    if (!hasO) missingElements.push('Oxygen (O)');

    if (missingElements.length > 0) {
        // Special Case: Sodium Niobate (NaNbO3) which has Nb, O, Na but no K
        if (!hasK && hasNa && hasNb && hasO) {
            return {
                isValid: false,
                errorTitle: "Scope Limitation",
                userMessage: "Pure Sodium Niobate (NaNbO3) is excluded. We are currently focusing on KN and KNN based systems. We are currently expanding our model capability to include other types of piezoelectric materials and enhance prediction accuracy."
            };
        }

        const missingStr = missingElements.join(', ');
        return {
            isValid: false,
            errorTitle: "Missing Critical Elements",
            userMessage: `To be classified as a KN or KNN-based ceramic, the formula MUST contain ${missingStr}. We are currently focusing on KN and KNN based systems and currently working to expand our model capability to include other types of piezoelectric materials and enhance prediction accuracy.`
        };
    }

    // Classification (Internal Info / Success)
    // Both KN-based (No Na) and KNN-based (With Na) are allowed.
    // The validation passes here.

    return { isValid: true, errorTitle: null, userMessage: null };
};
