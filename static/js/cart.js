// Initialize cart
let cart = JSON.parse(localStorage.getItem('cart')) || [];

function addToCart(name, price, category) {
    let item = cart.find(i => i.name === name);
    if (item) {
        item.quantity += 1;
    } else {
        cart.push({ name, price, category, quantity: 1 });
    }
    updateCart();
}

function updateCart() {
    localStorage.setItem('cart', JSON.stringify(cart));
    document.getElementById('cart-count').innerText = cart.reduce((acc, item) => acc + item.quantity, 0);
    displayCart();
}

function displayCart() {
    let cartList = document.getElementById('cart-items');
    let total = 0;
    cartList.innerHTML = "";
    
    cart.forEach((item, index) => {
        total += item.price * item.quantity;
        cartList.innerHTML += `
            <li>
                ${item.name} (x${item.quantity}) - ₹${item.price * item.quantity}
                <button onclick="removeFromCart(${index})">❌</button>
            </li>`;
    });

    document.getElementById('total-price').innerText = total;
}

function removeFromCart(index) {
    cart.splice(index, 1);
    updateCart();
}

function openCart() {
    document.getElementById('cart-sidebar').style.right = "0";
}

function closeCart() {
    document.getElementById('cart-sidebar').style.right = "-300px";
}

function checkout() {
    alert("Proceeding to checkout!");
    cart = [];
    updateCart();
}

// Load cart on page load
document.addEventListener("DOMContentLoaded", updateCart);
